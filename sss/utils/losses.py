"""Loss function wrappers, which are used with training pipelines.
"""

import tensorflow as tf


def cross_entropy(labels, logits, weights):
    """Compute cross entropy loss per pixel and reduce mean.

    Parameters
    ----------
    labels: (H, W, 1) tf.Tensor
        A ground truth label image where
        each pixel contains true class id.
    logits: (H, W, C) tf.Tensor
        A predicted logits values.
    weights: (H, W) tf.Tensor
        weights which weight on losses of each pixel
        before conducting reduce mean operation.
    """
    return tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(
            labels=tf.squeeze(labels, squeeze_dims=[3]),
            logits=logits,
            weights=weights))
