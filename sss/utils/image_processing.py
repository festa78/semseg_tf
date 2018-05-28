"""Image processing utils.
"""

import tensorflow as tf


def random_crop_image_and_label(image, label, crop_size):
    """Randomly crops `image` together with `label`.
    cf.https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way

    Parameters
    ----------
    image: (H, W, C=3) tf.Tensor
        image tensor.
    label: (H, W, C=1) tf.Tensor
        label tensor.
    crop_size: (H, W) 1D tf.Tensor
        Height (H) and Width (W) of crop size.

    Returns
    -------
    image_crop: (H, W, C=3) tf.Tensor.
        Cropped image tensor.
    label_crop: (H, W, C=1) tf.Tensor.
        Cropped label tensor.
    """
    # Concatenate along the channel axis.
    label = tf.cast(label, tf.float32)
    combined = tf.concat((image, label), axis=2)
    combined_crop = tf.random_crop(
        combined, size=(crop_size[0], crop_size[1], 4))

    image_crop, label_crop = combined_crop[..., :3], combined_crop[..., 3]
    label_crop = tf.cast(label_crop, tf.int64)
    # Avoid losing shape information.
    image_crop = tf.reshape(image_crop, (crop_size[0], crop_size[1], 3))
    label_crop = tf.reshape(label_crop, (crop_size[0], crop_size[1], 1))

    return image_crop, label_crop
