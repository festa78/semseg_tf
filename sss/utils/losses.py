"""Loss function wrappers, which are used with training pipelines.
"""

import tensorflow as tf


def cross_entropy(predictions, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=tf.squeeze(labels, squeeze_dims=[3])))
