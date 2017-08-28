import tensorflow as tf
import numpy as np


"""
Implmenentaion of ABCNNs
(https://arxiv.org/pdf/1512.05193.pdf)

:param s: sentence length
:param w: filter width
:param l2_reg: L2 regularization coefficient
:param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
:param d0: dimensionality of word embedding(default: 300)
:param di: The number of convolution kernels (default: 50)
:param num_layers: The number of convolution layers.
:param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
:param num_classes: The number of classes for answers.
"""

# zero padding to inputs for wide convolution
def pad_for_wide_conv(x, w):
    return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
    return dot_products / (norm1 * norm2)

def euclidean_score(v1, v2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
    return 1 / (1 + euclidean)

def make_attention_mat(x1, x2):
    # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
    # x2 => [batch, height, 1, width]
    # [batch, width, wdith] = [batch, s, s]
    return euclidean_score(x1, tf.matrix_transpose(x2))

def convolution(name_scope, x, d, reuse, hparams):
    with tf.name_scope(name_scope + "-conv"):
        with tf.variable_scope("conv") as scope:
            conv = tf.contrib.layers.conv2d(
                inputs=x,
                num_outputs=hparams.di,
                kernel_size=(d, hparams.w),
                stride=1,
                padding="VALID",
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=hparams.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                reuse=reuse,
                trainable=True,
                scope=scope
            )
            # Weight: [filter_height, filter_width, in_channels, out_channels]
            # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

            # [batch, di, s+w-1, 1]
            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
            return conv_trans

def w_pool(variable_scope, x, attention, hparams):
    # x: [batch, di, s+w-1, 1]
    # attention: [batch, s+w-1]
    model_type = hparams.model_type
    with tf.variable_scope(variable_scope + "-w_pool"):
        if model_type == "ABCNN2" or model_type == "ABCNN3":
            pools = []
            # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
            for i in range(hparams.s):
                # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                pools.append(tf.reduce_sum(x[:, :, i:i + hparams.w, :] * attention[:, :, i:i + hparams.w, :],
                                           axis=2,
                                           keep_dims=True))
            # [batch, di, s, 1]
            w_ap = tf.concat(pools, axis=2, name="w_ap")
        else:
            # [batch, di, s, 1]
            w_ap = tf.layers.average_pooling2d(
                inputs=x,
                # (pool_height, pool_width)
                pool_size=(1, hparams.w),
                strides=1,
                padding="VALID",
                name="w_ap"
            )
        return w_ap

def all_pool(variable_scope, x, pool_width, d):
    with tf.variable_scope(variable_scope + "-all_pool"):
        # [batch, di, 1, 1]
        all_ap = tf.layers.average_pooling2d(
            inputs=x,
            # (pool_height, pool_width)
            pool_size=(1, pool_width),
            strides=1,
            padding="VALID",
            name="all_ap"
        )
        # [batch, di]
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])
        #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

        return all_ap_reshaped

def CNN_layer(variable_scope, x1, x2, d, hparams):
    model_type = hparams.model_type
    # x1, x2 = [batch, d, s, 1]
    with tf.variable_scope(variable_scope):
        if model_type == "ABCNN1" or model_type == "ABCNN3":
            with tf.name_scope("att_mat"):
                aW = tf.get_variable(name="aW",
                                     shape=(hparams.s, d),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=hparams.l2_reg))

                # [batch, s, s]
                att_mat = make_attention_mat(x1, x2)

                # [batch, s, s] * [s,d] => [batch, s, d]
                # matrix transpose => [batch, d, s]
                # expand dims => [batch, d, s, 1]
                x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)

                # [batch, d, s, 2]
                x1 = tf.concat([x1, x1_a], axis=3)
                x2 = tf.concat([x2, x2_a], axis=3)

        left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1, hparams.w), d=d, reuse=False, hparams=hparams)
        right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2, hparams.w), d=d, reuse=True, hparams=hparams)

        left_attention, right_attention = None, None

        if model_type == "ABCNN2" or model_type == "ABCNN3":
            # [batch, s+w-1, s+w-1]
            att_mat = make_attention_mat(left_conv, right_conv)
            # [batch, s+w-1], [batch, s+w-1]
            left_attention = tf.reduce_sum(att_mat, axis=2)
            right_attention = tf.reduce_sum(att_mat, axis=1)

        pool_width = hparams.s + hparams.w - 1
        left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention, hparams=hparams)
        left_ap = all_pool(variable_scope="left", x=left_conv, pool_width=pool_width, d=hparams.di)
        right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention, hparams=hparams)
        right_ap = all_pool(variable_scope="right", x=right_conv, pool_width=pool_width, d=hparams.di)

        return left_wp, left_ap, right_wp, right_ap

def get_embeddings(hparams):
  if hparams.glove_path and hparams.vocab_path:
    tf.logging.info("Loading Glove embeddings...")
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
  else:
    tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.d0],
    initializer=initializer)


def abcnn_model(hparams,
                mode,
                context,
                context_len,
                utterance,
                utterance_len,
                targets):

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")
  with tf.variable_scope("cnn") as vs:
    x1 = tf.transpose(context_embedded, perm=[0, 2, 1])
    x2 = tf.transpose(utterance_embedded, perm=[0, 2, 1])
    x1_expanded = tf.expand_dims(x1, -1)
    x2_expanded = tf.expand_dims(x2, -1)


    LO_0 = all_pool(variable_scope="input-left", x=x1_expanded, pool_width=hparams.s, d=hparams.d0)
    RO_0 = all_pool(variable_scope="input-right", x=x2_expanded, pool_width=hparams.s, d=hparams.d0)

    LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=hparams.d0, hparams=hparams)
    out_left, out_right = LO_1, RO_1
    # sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]

    _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=hparams.di, hparams=hparams)
    out_left, out_right = LO_2, RO_2
    # sims.append(cos_sim(LO_2, RO_2))

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M",
      shape=[hparams.di, hparams.di],
      initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.matmul(out_left, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_utterance = tf.expand_dims(out_right, 2)

    # Dot product between generated response and actual response
    # (c * M) * r
    logits = tf.matmul(generated_response, encoding_utterance, adjoint_a=True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
