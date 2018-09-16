"""Loss function wrappers, which are used with training pipelines.
"""

import tensorflow as tf


def _ignore_zero_weight(labels, logits, weights):
    ignore_mask = tf.greater(weights, 0.)
    labels = tf.multiply(labels, tf.cast(ignore_mask, labels.dtype))
    logits = tf.multiply(logits, tf.cast(ignore_mask, logits.dtype))

    return labels, logits


def cross_entropy(labels, logits, weights):
    """Compute cross entropy loss per pixel and reduce mean.

    Parameters
    ----------
    labels: (N, H, W, C=1) tf.Tensor
        A ground truth label image where
        each pixel contains true class id.
    logits: (N, H, W, C) tf.Tensor
        A predicted logits values.
    weights: (N, H, W, C=1) tf.tensor
        Weights tensor to compute weighted loss.
        If weight is 0, the corresponding element is ignored.
        This has the same shape to @p labels.
    ignore_id: int, default 255
        Set weight as 0 if class id is @p ignore_id.
        This means it will ignore losses of @p ignore_id class.

    Returns
    -------
    Scalar tensor of mean cross entropy loss.
    """
    labels, logits = _ignore_zero_weight(labels, logits, weights)
    return tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(
            labels=tf.squeeze(labels, [3]), logits=logits, weights=weights))
