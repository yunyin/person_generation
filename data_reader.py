# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np

class DataReader():
  def __init__(self, datafiles, metadata, seq_length = 1, batch_size = 1):
    self.seq_length = seq_length
    self.batch_size = batch_size

    self.data = []
    for line in open(datafiles):
      line = line.strip().replace(" ", "")
      words = list(line.decode('utf-8'))
      words.append("</s>")
      words.insert(0, "<s>")
      self.data.append(words)

    self.words = list(set(sum(self.data, [])))
    self.words.sort()
    # vocabulary
    self.vocab_size = len(self.words)  # vocabulary size
    print 'Vocabulary Size: %d' % self.vocab_size
    self.char2id_dict = {w: i for i, w in enumerate(self.words)}
    self.id2char_dict = {i: w for i, w in enumerate(self.words)}

    self.data = self.arrange_data(self.data, self.batch_size)
    # pointer position to generate current batch
    self.reset()

    # save metadata file
    self.save_metadata(metadata)

  def arrange_data(self, data, batch_size):
    batch_len = np.array([0] * batch_size)

    refine_data = []
    for i in range(batch_size): refine_data.append([])
    #refine_data = [[]] * batch_size
    for sen in data:
      idx = np.argmin(batch_len)
      refine_data[idx].extend(sen)
      batch_len[idx] += len(sen)

    return refine_data

  def reset(self):
    self.pointers = [0] * self.batch_size

  def char2id(self, c):
    return self.char2id_dict[c]

  def id2char(self, id):
    return self.id2char_dict[id]

  def save_metadata(self, file):
    with open(file, 'w') as f:
      f.write('id\tchar\n')
      for i in range(self.vocab_size):
        c = self.id2char(i)
        f.write('{}\t{}\n'.format(i, c.encode('utf-8')))

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
      bx = [self.char2id(c) for c in bx]
      by = [self.char2id(c) for c in by]
      x_batches.append(bx)
      y_batches.append(by)
      masks.append(b_mask)
      if self.pointers[i] < len(self.data[i]):
        self.pointers[i] += self.seq_length

    return x_batches, y_batches, masks

if __name__=='__main__':
  reader = DataReader(datafiles = sys.argv[1],
                      metadata = sys.argv[2],
                      batch_size = 2,
	              seq_length = 5)
  while reader.has_next_batch():
    x, y, m = reader.next_batch()
    print 'x', x
    print 'y', y
    print 'm', m
    print '====='
