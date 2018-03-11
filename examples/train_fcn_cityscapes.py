#!/usr/bin/python3 -B

import argparse

import tensorflow as tf


class FcnCityscapes:
    """Train a fully convolutional network on Cityscapes datasets.
    """

    def __init__(self, options):
        self.options = options
        self.dataset = []

    def _load_dataset(self):
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
            dataset = self._load_dataset()

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
    parser.add_argument(
        'data_path',
        type=str,
        default='~/workspace/local_data/cityscapes',
        help='Path to the cityscapes data directory.')
    options = parser.parse_args()

    cls = FcnCityscapes(options)
    cls.main()
