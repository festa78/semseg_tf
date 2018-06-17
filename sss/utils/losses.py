"""Loss function wrappers, which are used with training pipelines.
"""

import tensorflow as tf


def cross_entropy(labels, logits, weights):
    """Compute cross entropy loss per pixel and reduce mean.

    Parameters
    ----------
    labels: (N, H, W) tf.Tensor
        A ground truth label image where
        each pixel contains true class id.
    logits: (N, H, W, C) tf.Tensor
        A predicted logits values.
    weights: (N, H, W) tf.Tensor
        weights which weight on losses of each pixel
        before conducting reduce mean operation.

    Returns
    -------
    Scalar tensor of mean cross entropy loss.
    """
    return tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=weights))
