import os
import nibabel as nib
import gzip

import numpy as np
import tensorflow as tf

import keras
# tf.python.control_flow_ops = tf

from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose
from keras.layers import AveragePooling2D, UpSampling2D, merge, Dropout, MaxPooling2D
from keras.layers import Activation, Cropping2D, ZeroPadding2D, RepeatVector
from keras.layers import Conv3D, Conv3DTranspose, Cropping3D, ZeroPadding3D
from keras.layers import AveragePooling3D, UpSampling3D, MaxPooling3D
from keras.layers import Subtract, Multiply, Lambda, LeakyReLU, ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import Concatenate, concatenate
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.utils import to_categorical

from tensorflow.python.ops import nn

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def print_options(arg_dict):
    print('\n======================Training Options======================')
    for idx in arg_dict:
        print(idx+': '+str(arg_dict[idx]))
    print('\n') 
    
          
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef(y_true, y_pred, ismetric=True, smooth=1e-7):
    
    label_num = y_true.shape[-1]
    if ismetric:
        y_true = tf.cast(K.argmax(y_true, axis=3), tf.float32)
        y_pred = tf.cast(K.argmax(y_pred, axis=3), tf.float32)
        y_true = tf.cast(K.greater_equal(y_true, 2), tf.float32)
        y_pred = tf.cast(K.greater_equal(y_pred, 2), tf.float32)
    else:
        y_pred = y_pred[:,:,:,-1]
        y_true = y_true[:,:,:,-1]
    
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
        
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred, ismetric = False) + nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred)
