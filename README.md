TensorFlow-based semantic segmentation codes.

## Implemented model and data I/O.

### Model
- [FCN](https://arxiv.org/abs/1411.4038) model.
- [Dilated Network](https://arxiv.org/abs/1511.07122) model.

### Data I/O
- [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

## Prepare .tfrecord data
This repo only assume .tfrecord format data.  
The .tfrecord data should be created by `scripts/${data_name}_write.py`.  

## Training
You can train model by `scripts/${model_name}_${data_name}_trainer.py`.  
Example parameters are defined in `scripts/params/${model_name}_${data_name}_train_params.yaml`.  

## Prediction
You can run model inference by `scripts/${model_name}_${data_name}_predictor.py`.  
Example parameters are defined in `scripts/params/${model_name}_${data_name}_predict_params.yaml`.  
