# Dilated Saliency U-Net
Tensorflow-Keras Dilated Saliency UNet code
DSU-Net implemented to segment White Matter Hyperintensities on brain MR images. 

# Data Preparation
1. You must make a directory for training and test datasets like the 'data/com_test_configs_2fold_adni60' directory.
2. The directory must contain list files of CSF, FLIAR, IAM, ICV, T1w and WMH(label). 

# How to Train 

Run `main.py` file to train U-Net, Saliency U-Net or Dilated Saliency U-Net with your choice of data. 

The example of each model is described in `main.py` file. 

Train options (e.g. epoch, learning rate ...) can be changed in `utils.py` file (Please, see the `set_parser` section).

'--gpu_device' must be set with the available GPU number. 

## Linux command line example
```
  $ mkdir results
  $ python3 main.py --gpu_device 2 --depth 1 --num_epochs 80 --fold 1 --lr 1e-5 --reduce_lr_factor 0.5 --img_size 64
```
# Publication
This work has been published in _Frontiers in aging neuroscience_ 

[Jeong, Yunhee, et al. "Dilated saliency u-net for white matter hyperintensities segmentation using irregularity age map." Frontiers in aging neuroscience 11 (2019): 150.](https://doi.org/10.3389/fnagi.2019.00150)
