import os
import sys
import tensorflow as tf
import udc_inputs
import udc_model
import udc_metrics
import rnn_hparams
import cnn_hparams
from models.dual_encoder import dual_encoder_model
from models.abcnn import abcnn_model

tf.flags.DEFINE_string("input_dir", "./data", "Path of test data in TFRecords format")
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("model", "rnn", "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("test_batch_size", 8, "Batch size for testing")
FLAGS = tf.flags.FLAGS

TEST_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, FLAGS.model, "test.tfrecords"))

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

tf.logging.set_verbosity(FLAGS.loglevel)

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

  tf.reset_default_graph()
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig())
  input_fn_test = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[TEST_FILE],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1)
  eval_metrics = udc_metrics.create_evaluation_metrics()
  tf.reset_default_graph()
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
