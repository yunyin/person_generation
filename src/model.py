# -*- coding:utf-8 -*-
import os
import sys
import time

import inspect
import numpy as np
import tensorflow as tf

class Model():
  def lstm_cell(self):
    if 'reuse' in inspect.getargspec(
        tf.contrib.rnn.BasicLSTMCell.__init__).args:
      return tf.contrib.rnn.BasicLSTMCell(
          self.config['hidden_size'], forget_bias = 0.0, state_is_tuple = True,
          reuse = tf.get_variable_scope().reuse)
    else:
      return tf.contrib.rnn.BasicLSTMCell(
          self.config['hidden_size'], forget_bias = 0.0, state_is_tuple = True)

  def __init__(self, is_training, config, optimizer = None, lr = None):
    self.config = config
    seq_length = config['seq_length']
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']

    self._optimizer = optimizer
    self._lr = lr

    with tf.name_scope('inputs'):
      self.input_data = tf.placeholder(tf.int32, [None, seq_length], name = 'input_data')
      self.target_data = tf.placeholder(tf.int32, [None, seq_length], name = 'target_data')
      self.mask = tf.placeholder(config['data_type'], [None, seq_length], name = 'mask')

    with tf.name_scope('model'):
      basic_cell = self.lstm_cell
      self.cell = tf.contrib.rnn.MultiRNNCell(
          [basic_cell()] * config['num_layers'], state_is_tuple = True)
      ibatch = tf.shape(self.input_data)[0]
      self._initial_state = self.cell.zero_state(ibatch, config['data_type'])

      with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, hidden_size], dtype = config['data_type'])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

      outputs = []
      state = self._initial_state
      with tf.variable_scope("RNN"):
        for time_step in range(seq_length):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          (cell_output, state) = self.cell(inputs[:, time_step, :], state)
          outputs.append(cell_output)

    with tf.name_scope('loss'):
      output = tf.reshape(tf.stack(axis = 1, values = outputs), [-1, hidden_size])
      softmax_w = tf.get_variable(
          "softmax_w", [hidden_size, vocab_size], dtype = config['data_type'])
      softmax_b = tf.get_variable(
          "softmax_b", [vocab_size], dtype = config['data_type'])
      logits = tf.matmul(output, softmax_w) + softmax_b

      logits = tf.reshape(logits, [-1, seq_length, vocab_size])

      self.probs = tf.nn.softmax(logits, name = 'probs')
      self._final_state = state

      self.one_hot = tf.one_hot(indices = self.target_data,
                                depth = vocab_size,
                                on_value = 1.,
                                off_value = 0.,
                                axis = -1)
      self.target_probs = self.one_hot * self.probs
      self.target_probs = tf.reduce_sum(self.target_probs, axis = 2)

      if not is_training: return
      loss = tf.contrib.seq2seq.sequence_loss(
          logits = logits,
          targets = self.target_data,
          weights = self.mask,
          average_across_timesteps=False,
          average_across_batch=True)

      self._cost = cost = tf.reduce_sum(loss)

    with tf.name_scope('optimize'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config['max_grad_norm'])
      self._train_op = self._optimizer.apply_gradients(zip(grads, tvars),
                 global_step = tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(tf.float32, shape=[])
    self._lr_update = tf.assign(self._lr, self._new_lr)

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def train_op(self):
    return self._train_op

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def lr(self):
    return self._lr

  def assign_lr(self, sess, new_lr):
    sess.run(self._lr_update, feed_dict={self._new_lr: new_lr})
