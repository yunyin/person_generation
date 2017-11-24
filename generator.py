# -*- coding:utf-8 -*-
import os
import sys
import time

import numpy as np
import tensorflow as tf

import data_reader
import model
import optimizer

flags = tf.flags
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
FLAGS = flags.FLAGS

class HParam(config):
  batch_size = 1
  n_epoch = 5
  learning_rate = 0.01
  decay_steps = 1000
  decay_rate = 0.9
  max_grad_norm = 5
  grad_clip = 5
  data_type = tf.float32

  vocab_size = 52
  hidden_size = 100
  num_layers = 1
  seq_length = 10
  metadata = 'metadata.tsv'

def run_epoch(sess, model, data, config, is_training = False):
  data.reset()
  costs = 0
  iters = 0
  while data.has_next_batch():
    x_batch, y_batch, mask = data.next_batch()
    feed_dict = {model.input_data: x_batch,
                 model.target_data: y_batch,
                 model.lr: config.learning_rate}
    fetches = {"cost": model._cost,
               "final_state": model._final_state}
    if is_training: fetches["train_op"] = model.train_op
    vals = sess.run(fetches, feed_dict)
    costs += vals["cost"]
    iters += data.batch_size
  return costs, iters

def train(config):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path for data")

  data = data_reader.DataReader(FLAGS.data_path, \
                                config.metadata, \
                                batch_size = config.batch_size,
                                seq_length = config.seq_length)

  opt, _ = optimizer.get_optimizer("sgd", config.learning_rate)
  train_model = model.Model(is_training = True, \
                            config = config, \
                            optimizer = opt)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(config.n_epoch):
      costs, iters = run_epoch(sess, train_model, data, \
			       learning_rate = config.learning_rate, \
			       is_training = True)
      print 'Iter {}: training_loss:{:4f}'.format(i, costs / iters)

def main(_):
  config = HParam()
  train(config)

if __name__ == '__main__':
  tf.app.run()
