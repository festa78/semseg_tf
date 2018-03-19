import tensorflow as tf


class BaseTFRecordWriter:
    """Base class to write .tfrecord file for semantic segmentation.
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def _write_tfrecord(self):
        # XXX: Need some assertions to self.data_list.
        print(self.data_list['test']['image_list'][:10])
        print(self.data_list['test']['label_list'][:10])
