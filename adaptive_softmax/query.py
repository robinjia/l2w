"""Insert a description of this module."""
import argparse
import json
import numpy as np
import os
import sys
from tqdm import tqdm
import torch
import torchfile

import util

PARAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'l2w-model')

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('in_file')
  parser.add_argument('--neighbors-file', '-n')
  parser.add_argument('--window-radius', '-w', type=int, default=6)
  parser.add_argument('--batch-size', '-b', type=int, default=1)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

class QueryHandler():
  def __init__(self, model, word_to_idx, mapto, device):
    self.model = model
    self.word_to_idx = word_to_idx
    self.mapto = mapto
    self.device = device

  def query(self, sentences, batch_size=1):
    T = len(sentences[0])
    if any(len(s) != T for s in sentences):
      raise ValueError('Only same length batches are allowed')

    log_probs = []
    for start in range(0, len(sentences), batch_size):
      batch = sentences[start:min(len(sentences), start + batch_size)]
      raw_idx_list = [[] for i in range(T+1)]
      for i, s in enumerate(batch):
        words = ['<S>'] + s
        word_idxs = [self.word_to_idx[w] for w in words]
        for t in range(T+1):
          raw_idx_list[t].append(word_idxs[t])
      all_raw_idxs = torch.tensor(raw_idx_list, device=self.device,
                                  dtype=torch.long)
      word_idxs = self.mapto[all_raw_idxs]
      hidden = self.model.init_hidden(len(batch))
      source = word_idxs[:-1,:]
      target = word_idxs[1:,:]
      decode, hidden = self.model(source, hidden)
      decode = decode.view(T, len(batch), -1)
      for i in range(len(batch)):
        log_probs.append(sum([decode[t, i, target[t, i]].item() for t in range(T)]))
    return log_probs


def load_model(device):
  word_map = torchfile.load(os.path.join(PARAM_DIR, 'word_map.th7'))
  word_map = [w.decode('utf-8') for w in word_map]
  word_to_idx = {w: i for i, w in enumerate(word_map)}
  word_freq = torchfile.load(os.path.join(os.path.join(PARAM_DIR, 'word_freq.th7')))
  mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long().to(device)

  with open(os.path.join(PARAM_DIR, 'lm.pt'), 'rb') as model_file:
    model = torch.load(model_file)
  model.full = True  # Use real softmax--important!
  model.to(device)
  model.eval()
  return QueryHandler(model, word_to_idx, mapto, device)


def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_handler = load_model(device)
  with open(OPTS.in_file) as f:
    sentences = [line.strip().split() for line in f]
  if OPTS.neighbors_file:
    with open(OPTS.neighbors_file) as f:
      neighbors = json.load(f)
    for sent_idx, s in enumerate(tqdm(sentences)):
      for i, w in enumerate(s):
        if w in neighbors:
          options = [w] + neighbors[w]
          start = max(0, i - OPTS.window_radius)
          end = min(len(s), i + 1 + OPTS.window_radius)
          # Remove OOV words from prefix and suffix
          prefix = [x for x in s[start:i] if x in query_handler.word_to_idx]
          suffix = [x for x in s[i+1:end] if x in query_handler.word_to_idx]
          queries = []
          in_vocab_options = []
          for opt in options:
            if opt in query_handler.word_to_idx:
              queries.append(prefix + [opt] + suffix)
              in_vocab_options.append(opt)
            else:
              print('%d\t%d\t%s\t%s' % (sent_idx, i, opt, float('-inf')))
          log_probs = query_handler.query(queries, batch_size=OPTS.batch_size)
          for x, lp in zip(in_vocab_options, log_probs):
            print('%d\t%d\t%s\t%s' % (sent_idx, i, x, lp))
  else:
    # Testing mode 
    log_probs = query_handler.query(sentences, batch_size=OPTS.batch_size)
    for s, lp in zip(sentences, log_probs):
      print('%s: %.2f' % (s, lp))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

