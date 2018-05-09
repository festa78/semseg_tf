import os

import glob
import tensorflow as tf

import project_root

from sss.data.tfrecord.base import BaseTFRecordWriter


class CityscapesTFRecordWriter(BaseTFRecordWriter):
    """Make TFRecord datasets from row Cityscapes data.
    """
    # Constants.
    IMAGE_ROOT = 'leftImg8bit'
    LABEL_ROOT = 'gtFine'
    LABEL_SURFIX = '_labelIds.png'
    DATA_CATEGORY = ['train', 'val', 'test']

    def __init__(self, input_dir, output_dir):
        super().__init__(data_list={})
        # XXX: Need some assertions to check that input files properly exists.
        self.input_dir = os.path.abspath(os.path.expanduser(input_dir))
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))

    def get_file_path(self):
        """Cityscapes specific way to get file paths.
        """
        for category in self.DATA_CATEGORY:
            image_list = glob.glob(
                os.path.join(self.input_dir, self.IMAGE_ROOT, category,
                             '*/*'))

            # Get label path corresponds to each image path.
            label_list = []
            for f in image_list:
                root_path = os.path.join(self.input_dir,
                                         self.LABEL_ROOT, category)
                area_name = f.split('/')[-2]
                base_name = os.path.basename(f).replace(
                    self.IMAGE_ROOT, self.LABEL_ROOT).replace('.png', self.LABEL_SURFIX)
                label_list.append(os.path.join(root_path, area_name, base_name))

            self.data_list[category] = {
                'image_list': image_list,
                'label_list': label_list
            }

    def run(self):
        # Get {image, label} pair file path.
        self.get_file_path()

        # Write tfrecord file.
        self._write_tfrecord(self.output_dir)
