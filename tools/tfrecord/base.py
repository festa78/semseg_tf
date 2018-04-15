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

    def __init__(self, data_list={}):
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
                        'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
                    }))
                    writer.write(example.SerializeToString())
