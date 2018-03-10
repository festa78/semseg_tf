#!/usr/bin/python3 -B

import argparse

import tensorflow as tf

# Input flags.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', '~/workspace/local_data/cityscapes',
                           """Path to the cityscapes data directory.""")


class FcnCityscapes:
    """Train a fully convolutional network on Cityscapes datasets
    """

    def __init__(self, options):
        self.options = options

    def _create_dataset(self):
        return []

    def _preprocess_data(self):
        return []

    def _load_model(self):
        return []

    def _train(self):
        pass

    def main(self):
        with tf.Graph().as_default():
            # Load data.
            dataset = self._create_dataset()

            # Preprocess data.
            dataset = self._preprocess_data()

            # Load model.
            model = self._load_model()

            # Start train loop.
            self._train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Trainer for a fully convolutional network on Cityscapes dataset.")
    parser.add_argument('test')
    options = parser.parse_args()

    trainer = FcnCityscapes(options)
    trainer.main()
