import os
import sys
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import udc_inputs
import udc_model
import udc_metrics
import rnn_hparams
import cnn_hparams
from models.dual_encoder import dual_encoder_model
from models.abcnn import abcnn_model

tf.flags.DEFINE_string("input_dir", "./data", "Path of infer data in CSV format")
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("model", "rnn", "Directory to store model checkpoints (defaults to ./runs)")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

INPUT_DIR=os.path.join(FLAGS.input_dir, FLAGS.model)
OUTPUT_PATH=os.path.join(FLAGS.input_dir, FLAGS.model, 'output.json')

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  os.path.join(INPUT_DIR, 'vocab_processor.bin'))

# Load your own data here
# INPUT_CONTEXT = "Example context"
# POTENTIAL_RESPONSES = ["Response 1", "Response 2"]

df = pd.read_csv(os.path.join(INPUT_DIR, 'infer.csv'))
INPUT_CONTEXT_L = np.array(df['Answer'])
POTENTIAL_RESPONSES_L = np.array([df['Question']]+[df['Distractor_{}'.format(i)] for i in range(9)]).T

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  if FLAGS.model == "rnn":
    hparams = rnn_hparams.create_hparams()
    model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  elif FLAGS.model == "cnn":
    hparams = cnn_hparams.create_hparams()
    model_fn = udc_model.create_model_fn(hparams, model_impl=abcnn_model)
  else:
    print("invalid model")
    exit(1)

  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  ds = []
  for INPUT_CONTEXT, POTENTIAL_RESPONSES in zip(INPUT_CONTEXT_L, POTENTIAL_RESPONSES_L):
    a = POTENTIAL_RESPONSES[0]
    m = ''
    ap = 0
    maxp = 0
    for r in POTENTIAL_RESPONSES:
      prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
      prob = next(prob)[0]
      # print("{}: {:g}".format(r, prob))
      if r == a:
        ap = prob
      if maxp < prob:
        m = r
        maxp = prob

    ds.append({'max_prob': '{:g}'.format(maxp), 'true_prob':'{:g}'.format(ap),
               'max_q': m, 'true_q': a, 'match': 1 if a == m else 0})
    # print("True {}:{}".format(a, ap))
    # print("Max {}:{}".format(m, maxp))
  with open(OUTPUT_PATH, 'w') as fp:
    json.dump(ds, fp)
