import os
import sys
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import json
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import time

import data_reader

host = 'localhost'
port = '9000'
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

class time_control:
  def __init__(self):
    self.total_time = 0

  def start(self):
    self.start_time = time.time()

  def stop(self):
    self.end_time = time.time()
    self.total_time += self.end_time - self.start_time

  def get_total_time(self):
    return self.total_time

client_timer = time_control()
def client_batch(input, output, lens, batch_size, seq_length):
  # init request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'logits'
  request.model_spec.signature_name = 'predict'

  input_np = np.array(input, dtype = np.int32)
  output_np = np.array(output, dtype = np.int32)

  request.inputs['x'].CopyFrom(
      tf.contrib.util.make_tensor_proto(input_np,
                                        shape = [batch_size, seq_length]))
  request.inputs['y'].CopyFrom(
      tf.contrib.util.make_tensor_proto(output_np,
                                        shape = [batch_size, seq_length])
      )
  client_timer.start()
  result = stub.Predict(request, 10.0)
  client_timer.stop()

  probs = np.array(result.outputs['pred'].float_val).reshape((batch_size, seq_length))

  scores = [0] * batch_size
  for i in range(batch_size):
    for j in range(seq_length):
      if j >= lens[i]: continue
      scores[i] += np.log(probs[i][j])

  return scores

def eval(infile, outfile, config):
  # init vocab
  vocab = data_reader.Vocab(vocab_limits = config['vocab_size'])
  vocab.load_metadata(config['metadata'])
  config['vocab_size'] = vocab.vocab_size()
  tf.logging.info(config)

  seq_length = config['seq_length']
  out = open(outfile, 'w')

  index = None
  lens = []
  in_ids = []
  out_ids = []
  base = []

  total_timer = time_control()
  send_timer = time_control()
  total_timer.start()
  for line in open(infile):
    vec = line.strip().split()
    if index != vec[0]:
      if index != None:
        send_timer.start()
        scores = client_batch(input = in_ids, output = out_ids, lens = lens,
                              batch_size = len(in_ids), seq_length = seq_length)
        send_timer.stop()

        for i in range(len(in_ids)):
          if base[i] == None: break
          out.write('%s\t%d\t%.4f\n' % (base[i], lens[i], scores[i]))

      index = vec[0]
      lens = []
      in_ids = []
      out_ids = []
      base = []

    sen = list(''.join(vec[1:]).decode('utf-8'))
    sen.insert(0, "<s>")
    sen.append("</s>")
    lens.append(len(sen))
    sen.extend(["</s>"] * (seq_length - len(sen)))

    base.append(line.strip())
    in_ids.append([vocab.char2id(c) for c in sen])
    out_ids.append(in_ids[-1][1:])
    out_ids[-1].append(vocab.char2id("</s>"))

  send_timer.start()
  scores = client_batch(input = in_ids, output = out_ids, lens = lens,
                        batch_size = len(in_ids), seq_length = seq_length)
  send_timer.stop()

  for i in range(len(in_ids)):
    if base[i] == None: break
    out.write('%s\t%d\t%.4f\n' % (base[i], lens[i], scores[i]))

  total_timer.stop()
  print 'total_timer: %.2f' % total_timer.get_total_time()
  print 'send_timer: %.2f' % send_timer.get_total_time()
  print 'client_timer: %.2f' % client_timer.get_total_time()

if __name__ == "__main__":
  config = json.load(open(sys.argv[3]))
  if config['data_type'] == 'float16': config['data_type'] = tf.float16
  else: config['data_type'] = tf.float32
  config['seq_length'] = 100
  eval(infile = sys.argv[1], outfile = sys.argv[2], config = config)
