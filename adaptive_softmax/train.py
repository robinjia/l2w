import argparse
import sys, time, os
import math
import torch
import torch.nn as nn
import torch.optim as optim

path = os.path.realpath(__file__)
path = path[:path.rindex('/')+1]
sys.path.insert(0, os.path.join(path, '../utils/'))
from doing import doing

import corpus
import model

import util

import numpy as np

from gbw import GBWDataset
from fast_gbw import FastGBWDataset

from torch.utils.serialization import load_lua

from torch.autograd import Variable

from adaptive_softmax import AdaptiveLoss
from splitcross import SplitCrossEntropyLoss

import logging

logging.basicConfig(format='[%(asctime)s]: %(message)s',
                    datefmt='%m/%d %I:%M:%S %p', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--dic', type=str,
                    help='path to dictionary pickle')
parser.add_argument('--old', type=str, default=None,
                    help='old model to keep training')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, GRU)')
parser.add_argument('--emsize', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--proj', action='store_true',
                    help='flag true if nhid!=emsize')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--cutoffs', nargs='+', type=int,
                    help='cutoffs for buckets in adaptive softmax')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--ar', type=float, default=0.1,
                    help='learning rate annealing rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.01,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
# Hardware
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--lm1b', action='store_true',
                    help='use GBW training mode for training LM1B, including efficient data loading')
parser.add_argument('--valid_per_epoch', action='store_true',
                    help='only evaluate at the end of epoch')
parser.add_argument('--gpu', type=int,  default=0,
                    help='gpu to use')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)

if not args.lm1b:
    with doing('Loading data'):
        corpus = corpus.Corpus(args.data, args.dic)
        ntokens = len(corpus.dictionary.idx2word)
        cutoffs = args.cutoffs + [ntokens]
else:
    ###############################################################################
    # Load data
    ###############################################################################

    # Torch
    word_freq = load_lua(os.path.join(args.data, 'word_freq.th7')).numpy()
    mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long()
    print("load word frequency mapping - complete")

    ntokens = len(word_freq)
    nsampled = 8192

    train_corpus = FastGBWDataset(args.data, 'train_data.th7', 'train_data.sid', mapto)
    print("load train data - complete")

    test_corpus = GBWDataset(args.data, 'test_data.th7', mapto)
    print("load test data - complete")

    cutoffs = args.cutoffs + [ntokens]


# with doing('Constructing model'):
    # if not args.lm1b:
    #     criterion = AdaptiveLoss(cutoffs)
    # else:
    #     criterion = SplitCrossEntropyLoss(args.emsize, args.cutoffs, verbose=False)
    #     criterion.cuda()
logging.info("Constructing model")
criterion = AdaptiveLoss(cutoffs).cuda()
if args.old is None:
    logging.info("building model")
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, cutoffs, args.proj, args.dropout, args.tied,
                           args.lm1b)
else:
    with open(args.old, 'rb') as model_file:
        model = torch.load(model_file)
if args.cuda:
    model.cuda()

optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=1e-6)
eval_batch_size = 1


###############################################################################
# Training code
###############################################################################

# Loop over epochs.
global lr, best_val_loss
lr = args.lr
best_val_loss = None

def repackage_hidden(h):
    """Detaches hidden states from their history"""
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(item, device_id=0):
    data, target, wrd_cnt, batch_num = item
    return Variable(data.cuda(device_id)), Variable(target.view(-1).cuda(device_id)), wrd_cnt, batch_num

def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    global ntokens

    model.eval()
    total_loss, nbatches = 0, 0
    # ntokens = len(corpus.dictionary.idx2word) if not args.lm1b else ntokens
    hidden = model.init_hidden(args.eval_batch_size)

    if not args.lm1b:
        data_gen = corpus.iter(split, args.eval_batch_size, args.bptt, use_cuda=args.cuda)
    else:
        data_gen = test_corpus.batch_generator(seq_length=args.bptt, batch_size=eval_batch_size, shuffle=False)

    for item in data_gen:

        if args.lm1b:
            source, target, word_cnt, batch_num = get_batch(item)
        else:
            source, target = item

        model.softmax.set_target(target.data.view(-1))

        output, hidden = model(source, hidden)

        total_loss += criterion(output, target.view(-1)).data.sum()

        hidden = repackage_hidden(hidden)
        nbatches += 1
    return total_loss / nbatches


def train():
    global lr, best_val_loss
    # Turn on training mode which enables dropout.
    model.train()
    total_loss, nbatches = 0, 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)

    if not args.lm1b:
        data_gen = corpus.iter('train', args.batch_size, args.bptt, use_cuda=args.cuda)
    else:
        data_gen = train_corpus.batch_generator(seq_length=args.bptt, batch_size=args.batch_size)

    for b, batch in enumerate(data_gen):
        model.train()
        if args.lm1b:
            source, target, word_cnt, batch_len = get_batch(batch)
        else:
            source, target = batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()  # optimizer.zero_grad()
        model.softmax.set_target(target.data.view(-1))
        output, hidden = model(source, hidden)
        loss = criterion(output, target.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        # for p in model.parameters():
        #     if p.grad is not None:
        #         p.data.add_(-lr, p.grad.data)

        total_loss += loss.data.cpu()
        # logging.info(total_loss)

        if b % args.log_interval == 0 and b > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            if not args.valid_per_epoch:
                val_loss = evaluate('valid')
                logging.info('| epoch {:3d} | batch {:5d} | lr {:02.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                    epoch, b, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                    val_loss, math.exp(val_loss)))
            else:
                logging.info('| epoch {:3d} | batch {:5d} | lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} '.format(
                    epoch, b, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()


# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        logging.info("training on epoch {}".format(epoch))
        train()
        val_loss = evaluate('valid')

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        # else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # lr *= args.ar

        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
