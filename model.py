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

  def __init__(self, is_training, config, optimizer = None):
    self.config = config
    batch_size = config['batch_size']
    seq_length = config['seq_length']
    hidden_size = config['hidden_size']
    vocab_size = config['vocab_size']

    self._optimizer = optimizer

    with tf.name_scope('inputs'):
      self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
      self.target_data = tf.placeholder(tf.int32, [batch_size, seq_length])

    with tf.name_scope('model'):
      basic_cell = self.lstm_cell
      self.cell = tf.contrib.rnn.MultiRNNCell(
          [basic_cell()] * config['num_layers'], state_is_tuple = True)

      self._initial_state = self.cell.zero_state(batch_size, config['data_type'])

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
      self.probs = tf.nn.softmax(logits)

      logits = tf.reshape(logits, [batch_size, seq_length, vocab_size])

      loss = tf.contrib.seq2seq.sequence_loss(
          logits,
          self.target_data,
          tf.ones([batch_size, seq_length],
          dtype = config['data_type']),
          average_across_timesteps=False,
          average_across_batch=True)

      self._cost = cost = tf.reduce_sum(loss)
      self._final_state = state

    if not is_training: return

    with tf.name_scope('optimize'):
      self._lr = tf.placeholder(tf.float32, [])

      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config['max_grad_norm'])
      self._train_op = self._optimizer.apply_gradients(zip(grads, tvars),
                 global_step = tf.contrib.framework.get_or_create_global_step())

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
