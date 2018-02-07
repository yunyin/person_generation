# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np

class Vocab():
  def __init__(self, vocab_limits = -1):
    self._vocab_size = vocab_limits

  def create_vocab(self, datafiles, metadata, vocab_limits = -1):
    print 'Start Vocab Create'
    sys.stdout.flush()
    total_data = []
    for line in open(datafiles):
      words = line.strip().split()
      total_data.extend(words)

    print 'Data Load End For Vocab Create'
    sys.stdout.flush()
    words = list(set(total_data))
    words.sort()
    words.insert(0, '<unk>')
    words.insert(0, '</s>')
    words.insert(0, '<s>')

    if vocab_limits == -1: self._vocab_size = len(words)
    else: self._vocab_size = min(vocab_limits, len(words))
    words = words[:self._vocab_size]

    print 'Vocabulary Size: %d' % self._vocab_size
    sys.stdout.flush()
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

  def load_metadata(self, file):
    if not os.path.exists(file):
      print 'Vocab Metadata {} does not exists'.format(file)
      sys.exit(-1)
    self.char2id_dict = dict()
    self.id2char_dict = dict()

    cnt = 0
    for line in open(file):
      idx, word = line.strip().split('\t')
      self.char2id_dict[word.decode('utf-8')] = int(idx)
      self.id2char_dict[int(idx)] = word.decode('utf-8')
      cnt += 1
      if cnt == self._vocab_size: break
    self._vocab_size = len(self.id2char_dict)
    print 'Loading Vocabulary Size:{}'.format(self._vocab_size)
    sys.stdout.flush()

  def save_metadata(self, file):
    with open(file, 'w') as f:
      for i in range(self._vocab_size):
        c = self.id2char(i)
        f.write('{}\t{}\n'.format(i, c))

class DataReader():
  def __init__(self, datafiles, vocab, seq_length = 1, batch_size = 1):
    self.seq_length = seq_length
    self.batch_size = batch_size
    self.vocab = vocab

    print 'Start Read Data'
    sys.stdout.flush()
    self.data = []
    for line in open(datafiles):
      sen = line.strip().split('\t')[0]
      words = list(sen.replace(' ', '').decode('utf-8'))
      words.append("</s>")
      words.insert(0, "<s>")
      words = [self.vocab.char2id(c) for c in words]
      self.data.append(words)

    print 'Read Data End'
    sys.stdout.flush()

    self.data = self.arrange_data(self.data, self.batch_size)
    # pointer position to generate current batch
    self.reset()

  def arrange_data(self, data, batch_size):
    print 'Start arrange data'
    sys.stdout.flush()
    batch_len = np.array([0] * batch_size)
    np.random.shuffle(data)
    refine_data = []
    for i in range(batch_size): refine_data.append([])
    for sen in data:
      idx = np.argmin(batch_len)
      refine_data[idx].extend(sen)
      batch_len[idx] += len(sen)

    print 'arrange data End'
    sys.stdout.flush()
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
        bx.extend([self.vocab.char2id("</s>")] * self.seq_length)
        by.extend([self.vocab.char2id("</s>")] * self.seq_length)
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
        bx.extend([self.vocab.char2id("</s>")] * left)
        by.extend([self.vocab.char2id("</s>")] * left)
        for j in range(self.seq_length - left, self.seq_length):
          b_mask[j] = 0

      # convert to ids
      x_batches.append(bx)
      y_batches.append(by)
      masks.append(b_mask)
      if self.pointers[i] < len(self.data[i]):
        self.pointers[i] += self.seq_length

    return x_batches, y_batches, masks

if __name__=='__main__':
  vocab = Vocab()
  vocab.create_vocab(datafiles = sys.argv[1], \
                     metadata = sys.argv[2], \
                     vocab_limits = int(sys.argv[3]))
  vocab.load_metadata(sys.argv[2]) 
