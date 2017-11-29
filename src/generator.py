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
flags.DEFINE_string("config", None,
                    "Directory of Config File.")
flags.DEFINE_string("type", None,
                    "Run Type: train/gen")
flags.DEFINE_integer("task_id", 0,
                    "Task Id for Each workers")
flags.DEFINE_string("job_name", 'worker',
                    "Job Type: ps/worker")
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def HParam():
  if not FLAGS.config:
    raise ValueError("Must set --config")

  if not os.path.exists(FLAGS.config):
    raise ValueError("Config File %s Does Not Exist" % (FLAGS.config))

  config = json.load(open(FLAGS.config))
  if config['data_type'] == 'float16':
    config['data_type'] = tf.float16
  else: config['data_type'] = tf.float32
  return config

def run_epoch(sess, model, data, is_training = False, gen_model = None, vocab = None):
  data.reset()
  costs = 0
  iters = 0
  times = 0
  while data.has_next_batch():
    x_batch, y_batch, mask = data.next_batch()
    feed_dict = {model.input_data: x_batch,
                 model.target_data: y_batch,
                 model.mask: mask}

    fetches = {"costs": model._cost}
    if is_training: fetches["train_op"] = model.train_op

    vals = sess.run(fetches, feed_dict)
    costs += vals["costs"]
    
    iters += data.seq_length
    times += 1
    if times % 2000 == 100:
      tf.logging.info('step {}: training_loss:{:4f}'.format(times, np.exp(costs / iters)))
    if times % 20000 == 0 and gen_model != None and vocab != None:
      sample(sess, model = gen_model, vocab = vocab)

  if gen_model != None and vocab != None:
    sample(sess, model = gen_model, vocab = vocab)
  return costs, iters

def sample(sess, model, vocab, max_gen_len = 50):
  x_batch = np.zeros((1, 1))
  word = "<s>"
  state = sess.run(model.initial_state)

  output = []
  while word != '</s>' and len(output) < max_gen_len:
    x_batch[0][0] = vocab.char2id(word)
    feed_dict = {model.input_data: x_batch, \
                 model._initial_state: state}
    fetches = {"probs": model.probs,
               "state": model.final_state}
    vals = sess.run(fetches, feed_dict)
    p = vals['probs'][0][0]
    state = vals['state']
    word = vocab.id2char(np.argmax(p))
    if word != "</s>": output.append(word)

  line = " ".join(output)
  tf.logging.info('generate_len: {}'.format(len(output)))
  tf.logging.info(line.encode('utf-8'))

def train(config):
  # init for cluster
  cluster_conf = config["cluster_conf"]
  cluster = tf.train.ClusterSpec(cluster_conf)
  server = tf.train.Server(cluster,
                           job_name = FLAGS.job_name,
                           task_index = FLAGS.task_id)
  n_ps = len(cluster_conf['ps'])
  n_workers = len(cluster_conf['worker'])
  if FLAGS.job_name == "ps": server.join()
  is_chief = (FLAGS.task_id == 0)

  # load Vocab
  vocab = data_reader.Vocab(vocab_limits = config['vocab_size'])
  vocab.load_metadata(config['metadata'])
  config['vocab_size'] = vocab.vocab_size()
  tf.logging.info(config)

  # load Data
  train_data = data_reader.DataReader(config['train_data'][FLAGS.task_id],
                                      vocab = vocab,
                                      batch_size = config['batch_size'],
                                      seq_length = config['seq_length'])

  initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])

  # create models
  with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id,
                                                cluster = cluster)):
    with tf.name_scope('Train'):
      opt, lr = optimizer.get_optimizer("sgd", config['learning_rate'])
      sync_opt = tf.train.SyncReplicasOptimizer(opt,
                                                replicas_to_aggregate = n_workers * config['sync_freq'],
                                                total_num_replicas = n_workers)
      with tf.variable_scope("Model", reuse = None, initializer = initializer):
        train_model = model.Model(is_training = True, \
                                  config = config, \
                                  optimizer = sync_opt,
                                  lr = lr)

    with tf.name_scope('Generate'):
      generate_config = copy.deepcopy(config)
      generate_config['batch_size'] = 1
      generate_config['seq_length'] = 1
      with tf.variable_scope("Model", reuse = True, initializer = initializer):
        gen_model = model.Model(is_training = False, config = generate_config)

    chief_queue_runner = train_model.optimizer.get_chief_queue_runner()
    init_tokens_op = train_model.optimizer.get_init_tokens_op()

  sv = tf.train.Supervisor(is_chief = is_chief,
                           logdir = config['logdir'])
  sess_config = tf.ConfigProto(allow_soft_placement = True,
                               log_device_placement = True,
                               device_filters = ["/job:ps", "/job:worker/task:%d" % FLAGS.task_id])

  tf.logging.info('Start Sess')
  with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
    if is_chief:
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_tokens_op)

    for i in range(config['n_epoch']):
      lr_decay = config['lr_decay'] ** max(i + 1 - config['decay_epoch'], 0)
      train_model.assign_lr(sess, config['learning_rate'] * lr_decay)

      tf.logging.info('Iter {} Start, Learning_rate: {:4f}'.format(i, sess.run(train_model.lr)))
      costs, iters = run_epoch(sess, train_model, train_data,
                               is_training = True,
                               gen_model = gen_model,
                               vocab = vocab)
      tf.logging.info('Iter {}: training_loss:{:4f}'.format(i, np.exp(costs / iters)))

def gen(config):
  vocab = data_reader.Vocab(vocab_limits = config['vocab_size'])
  vocab.load_metadata(config['metadata'])
  config['vocab_size'] = vocab.vocab_size()
  tf.logging.info(config)

  with tf.name_scope('Generate'):
    config['batch_size'] = 1
    config['seq_length'] = 1
    with tf.variable_scope("Model", reuse = None):
      gen_model = model.Model(is_training = False, config = config)

  with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(config['logdir'])
    tf.logging.info(ckpt)
    saver.restore(sess, ckpt)

    sample(sess, model = gen_model, vocab = vocab)
  
def main(_):
  config = HParam()
  if not FLAGS.type:
    raise ValueError('run type is train/gen')
  if FLAGS.type == 'train':
    train(config)
  else:
    gen(config)

if __name__ == '__main__':
  tf.app.run()
