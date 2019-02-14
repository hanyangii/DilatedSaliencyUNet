# DilatedSaliencyUNet
Tensorflow-Keras Dilated Saliency UNet code

# Data Preperation
1. You have to make a directory for train and test dataset as 'data/com_test_configs_2fold_adni60' directory.
2. The directory must contain list files of CSF, FLIAR, IAM, ICV, T1w and WMH(label). 

# How to Train 
You have to run 'main.py' file to train U-Net, Saliency U-Net or Dilated Saliency U-Net with your choice of data. 

The example of each model is described in 'main.py' file. 

You can alter train options (e.g. epoch, learning rate ...) listed in 'set_parser' of 'utils.py' file.

'--gpu_device' must be set with the available GPU number. 

```
  $ mkdir results
  $ python3 main.py --gpu_device 2 --depth 1 --num_epochs 80 --fold 1 --lr 1e-5 --reduce_lr_factor 0.5 --img_size 64
```
