# -*- coding:utf-8 -*-
import os
import sys
import time

import numpy as np
import tensorflow as tf
import json

import copy
import data_reader
import model
import optimizer

flags = tf.flags
flags.DEFINE_string("train_data_path", None,
                    "Where the training data is stored.")
FLAGS = flags.FLAGS

def HParam():
  config = dict()
  config['batch_size'] = 32
  config['n_epoch'] = 5
  config['learning_rate'] = 0.001
  config['max_grad_norm'] = 5
  config['data_type'] = tf.float32
  config['init_scale'] = 0.1

  config['vocab_size'] = 7000
  config['hidden_size'] = 320
  config['num_layers'] = 2
  config['seq_length'] = 20
  config['metadata'] = 'metadata.tsv'
  return config

def run_epoch(sess, model, data, learning_rate, is_training = False):
  data.reset()
  costs = 0
  iters = 0
  times = 0
  while data.has_next_batch():
    x_batch, y_batch, mask = data.next_batch()
    feed_dict = {model.input_data: x_batch,
                 model.target_data: y_batch,
                 model.lr: learning_rate}
    fetches = {"cost": model._cost,
               "final_state": model._final_state}
    if is_training: fetches["train_op"] = model.train_op
    vals = sess.run(fetches, feed_dict)
    costs += vals["cost"]
    iters += data.batch_size
    times += 1
    if times % 2000 == 100:
      print 'step {}: training_loss:{:4f}'.format(times, np.exp(costs / iters))
      sys.stdout.flush()
  return costs, iters

def run_generate(sess, model, vocab, max_gen_len = 50):
  x_batch = np.zeros((1, 1))
  mask = np.ones((1, 1))
  word = "<s>"
  state = sess.run(model.cell.zero_state(1, tf.float32))

  output = []
  while word != '</s>' and len(output) < max_gen_len:
    x_batch[0][0] = vocab.char2id(word)
    feed_dict = {model.input_data: x_batch, \
                 model._initial_state: state}
    fetches = {"probs": model.probs,
               "state": model.final_state}
    vals = sess.run(fetches, feed_dict)
    p = vals['probs'][0]
    word = vocab.id2char(np.argmax(p))
    if word != "</s>": output.append(word)

  line = " ".join(output)
  print line.encode('utf-8')
  sys.stdout.flush()

def train(config):
  if not FLAGS.train_data_path:
    raise ValueError("Must set --train_data_path for data")

  vocab = data_reader.Vocab(FLAGS.train_data_path, \
                            config['metadata'], \
                            vocab_limits = config['vocab_size'])
  config['vocab'] = vocab.vocab_size()

  train_data = data_reader.DataReader(FLAGS.train_data_path, \
                                      vocab = vocab, \
                                      batch_size = config['batch_size'], \
                                      seq_length = config['seq_length'])
  initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
  with tf.name_scope('Train'):
    opt, _ = optimizer.get_optimizer("sgd", config['learning_rate'])
    with tf.variable_scope("Model", reuse = None, initializer = initializer):
      train_model = model.Model(is_training = True, \
                                config = config, \
                                optimizer = opt)

  with tf.name_scope('Generate'):
    generate_config = copy.deepcopy(config)
    generate_config['batch_size'] = 1
    generate_config['seq_length'] = 1
    with tf.variable_scope("Model", reuse = True, initializer = initializer):
      generate_model = model.Model(is_training = False, config = generate_config)

  print 'Start Sess'
  sys.stdout.flush()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(config['n_epoch']):
      print 'Iter {} Start'.format(i)
      costs, iters = run_epoch(sess, train_model, train_data, \
                               config['learning_rate'], \
                               is_training = True)
      print 'Iter {}: training_loss:{:4f}'.format(i, costs / iters)
      run_generate(sess, model = generate_model, vocab = vocab)

def main(_):
  config = HParam()
  train(config)

if __name__ == '__main__':
  tf.app.run()
