import tensorflow as tf
from collections import namedtuple

# Model Parameters
model_type="BCNN"
vocab_size=6929

embedding_dim = 250
hidden_dim = 50
window_size = 4
max_len = 40
l2_reg = 0.0004
# tf.flags.DEFINE_integer("num_layers", 2, "CNN layers")
# tf.flags.DEFINE_integer("max_context_len", 40, "Truncate contexts to this length")
# tf.flags.DEFINE_integer("max_utterance_len", 40, "Truncate utterance to this length")

# Pre-trained embeddings
glove_path=None
vocab_path=None

# Training Parameters
learning_rate = 0.001
batch_size = 64
eval_batch_size = 8
optimizer = "Adam"

HParams = namedtuple(
  "HParams",
  [
    "model_type",
    "vocab_size",
    "glove_path",
    "vocab_path",
    "d0",
    "di",
    "s",
    "w",
    "l2_reg",
    # "num_layers",
    "max_context_len",
    "max_utterance_len",
    "learning_rate",
    "batch_size",
    "eval_batch_size",
    "optimizer"
  ])

def create_hparams():
  return HParams(
    model_type=model_type,
    vocab_size=vocab_size,
    glove_path=glove_path,
    vocab_path=vocab_path,
    d0=embedding_dim,
    di=hidden_dim,
    s=max_len,
    w=window_size,
    l2_reg=l2_reg,
    # num_layers=FLAGS.num_layers,
    max_context_len=max_len,
    max_utterance_len=max_len,
    learning_rate=learning_rate,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    optimizer=optimizer
  )
