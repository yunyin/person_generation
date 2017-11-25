# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np

class Vocab():
  def __init__(self, datafiles, metadata, vocab_limits = -1):
    total_data = []
    for line in open(datafiles):
      words = list(line.strip().replace(" ", "").decode('utf-8'))
      total_data.extend(words)

    words = list(set(total_data))
    words.sort()
    words.insert(0, '<unk>')
    words.insert(0, '</s>')
    words.insert(0, '<s>')

    if vocab_limits == -1: self._vocab_size = len(words)
    else: self._vocab_size = min(vocab_limits, len(words))
    words = words[:self._vocab_size]

    print 'Vocabulary Size: %d' % self._vocab_size
    self.char2id_dict = {w: i for i, w in enumerate(words)}
    self.id2char_dict = {i: w for i, w in enumerate(words)}

    # save metadata file
    self.save_metadata(metadata)

  def vocab_size(self):
    return self._vocab_size

  def char2id(self, c):
    if not self.char2id_dict.has_key(c):
      c = '<unk>'
    return self.char2id_dict[c]

  def id2char(self, id):
    return self.id2char_dict[id]

  def save_metadata(self, file):
    with open(file, 'w') as f:
      f.write('id\tchar\n')
      for i in range(self._vocab_size):
        c = self.id2char(i)
        f.write('{}\t{}\n'.format(i, c.encode('utf-8')))

class DataReader():
  def __init__(self, datafiles, vocab, seq_length = 1, batch_size = 1):
    self.seq_length = seq_length
    self.batch_size = batch_size
    self.vocab = vocab

    self.data = []
    for line in open(datafiles):
      line = line.strip().replace(" ", "")
      words = list(line.decode('utf-8'))
      words.append("</s>")
      words.insert(0, "<s>")
      self.data.append(words)

    self.words = list(set(sum(self.data, [])))
    self.words.sort()

    self.data = self.arrange_data(self.data, self.batch_size)
    # pointer position to generate current batch
    self.reset()

  def arrange_data(self, data, batch_size):
    batch_len = np.array([0] * batch_size)

    refine_data = []
    for i in range(batch_size): refine_data.append([])
    for sen in data:
      idx = np.argmin(batch_len)
      refine_data[idx].extend(sen)
      batch_len[idx] += len(sen)

    return refine_data

  def reset(self):
    self.pointers = [0] * self.batch_size

  def has_next_batch(self):
    for idx in range(len(self.data)):
      if self.pointers[idx] < len(self.data[idx]) - 1: return True
    return False

  def next_batch(self):
    x_batches = []
    y_batches = []
    masks = []
    for i in range(self.batch_size):
      bx = []
      by = []
      b_mask = [1] * self.seq_length
      if self.pointers[i] >= len(self.data[i]) - 1:
        bx.extend(["</s>"] * self.seq_length)
        by.extend(["</s>"] * self.seq_length)
        b_mask = [0] * self.seq_length
      elif self.pointers[i] + self.seq_length + 1 < len(self.data[i]):
        bx.extend(self.data[i][self.pointers[i]:
                               self.pointers[i] + self.seq_length])
        by.extend(self.data[i][self.pointers[i] + 1:
                               self.pointers[i] + self.seq_length + 1])
      else:
        bx.extend(self.data[i][self.pointers[i]: len(self.data[i]) - 1])
        by.extend(self.data[i][self.pointers[i] + 1: len(self.data[i])])
        left = self.seq_length - (len(self.data[i]) - 1 - self.pointers[i])
        bx.extend(["</s>"] * left)
        by.extend(["</s>"] * left)
        for j in range(self.seq_length - left, self.seq_length):
          b_mask[j] = 0

      # convert to ids
      bx = [self.vocab.char2id(c) for c in bx]
      by = [self.vocab.char2id(c) for c in by]
      x_batches.append(bx)
      y_batches.append(by)
      masks.append(b_mask)
      if self.pointers[i] < len(self.data[i]):
        self.pointers[i] += self.seq_length

    return x_batches, y_batches, masks

if __name__=='__main__':
  vocab = Vocab(datafiles = sys.argv[1], metadata = sys.argv[2], vocab_limits = 50)
  reader = DataReader(datafiles = sys.argv[1],
                      vocab = vocab,
                      batch_size = 2,
                      seq_length = 5)
  while reader.has_next_batch():
    x, y, m = reader.next_batch()
    print 'x', x
    print 'y', y
    print 'm', m
    print '====='
