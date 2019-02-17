"""Insert a description of this module."""
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
import torch
import torchfile

import util

PARAM_DIR = 'l2w-model'

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('in_file')
  parser.add_argument('--rng-seed', type=int, default=123456)
  parser.add_argument('--torch-seed', type=int, default=1234567)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  word_map = torchfile.load(os.path.join(PARAM_DIR, 'word_map.th7'))
  word_map = [w.decode('utf-8') for w in word_map]
  word_to_idx = {w: i for i, w in enumerate(word_map)}
  word_freq = torchfile.load(os.path.join(os.path.join(PARAM_DIR, 'word_freq.th7')))
  mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long().to(device)

  with open(OPTS.in_file) as f:
    sentences = [line.strip().split() for line in f]
  with open(os.path.join(PARAM_DIR, 'lm.pt'), 'rb') as model_file:
    model = torch.load(model_file)
  model.full = True  # Use real softmax--important!
  model.to(device)
  model.eval()

  for s in tqdm(sentences):
    if not all(w in word_to_idx for w in s):
      print('OOV: %s from %s' % ([w for w in s if w not in word_to_idx], s), file=sys.stderr)
      continue
    input_words = ['<S>'] + s
    #input_words = s
    raw_idxs = torch.tensor([[word_to_idx[w]] for w in input_words], device=device, dtype=torch.long)
    word_idxs = mapto[raw_idxs]
    source = word_idxs[:-1,:]
    target = word_idxs[1:,:]
    hidden = model.init_hidden(1)
    #model.softmax.set_target(target.data.view(-1))
    decode, hidden = model(source, hidden)
    log_probs = [decode[t, target[t]].item() for t in range(len(s))]
    total_log_prob = sum(log_probs)
    print('%s: %.2f from %s' % (s, total_log_prob, log_probs))

if __name__ == '__main__':
  OPTS = parse_args()
  main()

