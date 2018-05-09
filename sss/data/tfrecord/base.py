import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import os
import PIL.Image as Image
import tensorflow as tf

import tqdm


class BaseTFRecordWriter:
    """Base class to write .tfrecord file for semantic segmentation.
    """

    def __init__(self, data_list):
        self.data_list = data_list
        self.logger = logging.getLogger(__name__)

    def _write_tfrecord(self, output_dir):
        for data_category in self.data_list.keys():
            filename = os.path.join(output_dir, data_category + '.tfrecord')
            self.logger.info('Start writing {} data to {}'.format(data_category, filename))
            with tf.python_io.TFRecordWriter(filename) as writer:
                for image_path, label_path in tqdm.tqdm(zip(self.data_list[data_category]['image_list'],
                                                            self.data_list[data_category]['label_list']),
                                                        total=len(self.data_list[data_category]['image_list'])):
                    # filename = os.path.basename(image_path)
                    filename = image_path
                    image = np.array(Image.open(image_path))
                    label = np.array(Image.open(label_path))
                    height = image.shape[0]
                    width = image.shape[1]
                    image_raw = image.tostring()
                    label_raw = label.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
                    }))
                    writer.write(example.SerializeToString())


class BaseTFRecordReader:
    """Base class to read .tfrecord file for semantic segmentation.
    This class works as an iterator.
    """
    def __init__(self, file_path, sess):
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path
        self.sess = sess
        self.dataset = tf.data.TFRecordDataset(self.file_path)
        self.dataset = self.dataset.map(self._parse_bytes_sample)

    def _parse_bytes_sample(self, bytedata):
        features = tf.parse_single_example(
        # serialized_example,
            bytedata,
        # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
                'filename': tf.FixedLenFeature([], tf.string)
        })

        filename = features['filename']

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.decode_raw(features['label_raw'], tf.uint8)

        height_org = tf.cast(features['height'], tf.int32)
        width_org = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height_org, width_org, 3])
        label_shape = tf.stack([height_org, width_org, 1])

        image = tf.reshape(image, image_shape)
        label = tf.reshape(label, label_shape)

        return image, label, filename

    def __iter__(self):
        iterator = self.dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __next__(self):
        try:
            image, label, filename = self.sess.run(self.next_element)
            return image, label, filename.decode()
        except tf.errors.OutOfRangeError:
            raise StopIteration
