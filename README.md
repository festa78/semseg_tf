TensorFlow-based semantic segmentation codes.

## Implemented models and data I/Os

### Model
- [Fully Convolutional Network](https://arxiv.org/abs/1411.4038) model.
- [Dilated Network](https://arxiv.org/abs/1511.07122) model.
- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) model.

### Data I/O
- [Cityscapes](https://www.cityscapes-dataset.com/) dataset.
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset.

## Prepare .tfrecord data
This repo only assumes .tfrecord format data.  
The .tfrecord data should be created by `scripts/write_tfrecord_${data_name}.py`.  

## Training
You can train model by `scripts/trainer_${model_name}_${data_name}.py`.  
Example parameters are defined in `scripts/params/train_params_${model_name}_${data_name}.yaml`.  

## Prediction
You can run model inference by `scripts/predictor_${model_name}_${data_name}.py`.  
Example parameters are defined in `scripts/params/predict_params_${model_name}_${data_name}.yaml`.  
