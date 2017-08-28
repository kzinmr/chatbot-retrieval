import os
import time
import tensorflow as tf
import udc_inputs
import udc_model
import udc_metrics
import rnn_hparams
import cnn_hparams
from models.dual_encoder import dual_encoder_model
from models.abcnn import abcnn_model

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model", "rnn", "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 100, "Evaluate after this many train steps")
FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, FLAGS.model, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, FLAGS.model, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):

  if FLAGS.model == "rnn":
    hparams = rnn_hparams.create_hparams()
    MODEL_DIR = os.path.abspath(os.path.join("./runs", FLAGS.model, str(TIMESTAMP)))
    model_fn = udc_model.create_model_fn(
      hparams,
      model_impl=dual_encoder_model)
  elif FLAGS.model == "cnn":
    hparams = cnn_hparams.create_hparams()
    MODEL_DIR = os.path.abspath(os.path.join("./runs", hparams.model_type, str(TIMESTAMP)))
    model_fn = udc_model.create_model_fn(
      hparams,
      model_impl=abcnn_model)
  else:
    print("invalid model")
    exit(1)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))
  input_fn_train = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)
  input_fn_eval = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)
  eval_metrics = udc_metrics.create_evaluation_metrics()
  eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

  estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])

if __name__ == "__main__":
  tf.app.run()
