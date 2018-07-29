TensorFlow-based semantic segmentation codes.

## Prepare .tfrecord data
This repo only assume .tfrecord format data.
The .tfrecord data should be created by `scripts/${data_name}_write.py`.
- Now only supports [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

## Training
You can training model by `scripts/${model_name}_${data_name}_trainer.py`.
- Now only implements [FCN](https://arxiv.org/abs/1411.4038) model.
Parameters are defined in `scripts/params/${model_name}_$data_name}_train_params.yaml`.

## Inference
WIP.
