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
    
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


def wasserstein_disagreement_map(
        prediction, ground_truth, weight_map=None, M=None):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened prediction and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    assert M is not None, "Distance matrix is required."
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    num_classes = prediction.shape[1].value
    ground_truth.set_shape(prediction.shape)
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(num_classes):
        for j in range(num_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in

        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score
        for Imbalanced Multi-class Segmentation using Holistic
        Convolutional Networks.MICCAI 2017 (BrainLes)

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    
    M_tree = np.array([[0., 1., 1., 1., 1.],
                       [1., 0., 0.6, 0.2, 0.5],
                       [1., 0.6, 0., 0.6, 0.7],
                       [1., 0.2, 0.6, 0., 0.5],
                       [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)

    
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    prediction = tf.cast(prediction, tf.float32)
    num_classes = prediction.shape[1].value
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    delta = wasserstein_disagreement_map(prediction, one_hot, M=M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :num_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL, dtype=tf.float32)


def GDL(y_true, y_pred, smooth=1e-7):
    #Weight map
    ref_vol = K.sum(y_true, axis=[0])
    intersect = K.sum(y_true * y_pred, axis=[0])
    seg_vol = K.sum(y_pred, axis=[0])
    #is_zero = tf.equal(ref_vol, tf.zeros_like(ref_vol))
    #ref_vol_where= tf.where(is_zero, tf.ones_like(ref_vol)*smooth, ref_vol)
    weights = tf.reciprocal(tf.square(ref_vol))
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights), weights)
    
    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))+smooth
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol))+smooth#tf.maximum(seg_vol + ref_vol, 2)))
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    
    #generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0, generalised_dice_score)
    return 1 - generalised_dice_score   
        
def iou_loss(y_true, y_pred):
    mul = y_pred*y_true
    mul = K.sum(mul, axis=(1,2))
    y_true = K.sum(y_true, axis=(1,2))
    y_pred = K.sum(y_pred, axis=(1,2))
    iou = K.sum(mul/(y_true+y_pred-mul), axis=-1)
    return -iou
    
def iou_crossentropy_loss(y_true, y_pred, axis=-1):
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(1e-3, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    denom = -tf.reduce_sum(y_true * tf.log(y_pred), (1,2)) + tf.reduce_sum(tf.log(y_pred), (1,2)) + tf.reduce_sum(y_true, (1,2))
    return - K.sum(tf.reduce_sum(y_true * tf.log(y_pred), (1,2))/denom)
    
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef(y_true, y_pred, ismetric=True, smooth=1e-7):
    
    label_num = y_true.shape[-1]
    if ismetric:
        y_true = tf.cast(K.argmax(y_true, axis=3), tf.float32)
        y_pred = tf.cast(K.argmax(y_pred, axis=3), tf.float32)
        #true_max = tf.cast(K.max(y_true), tf.float32)
        #pred_max = tf.cast(K.max(y_pred), tf.float32)
        y_true = tf.cast(K.greater_equal(y_true, 2), tf.float32)
        y_pred = tf.cast(K.greater_equal(y_pred, 2), tf.float32)
    else:
        y_pred = y_pred[:,:,:,-1]
        y_true = y_true[:,:,:,-1]
    
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
    
    #if ismetric and label_num>2:
    #    y_true_f = tf.cast(K.greater_equal(y_true_f, 2), tf.float32)
    #    y_pred_f = tf.cast(K.greater_equal(y_pred_f, 2), tf.float32)
        
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred, ismetric = False) + nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred)


def Saliency_Slice_UNet(layer_level, batch_norm, flair_shape, iam_shape, Loss, activation, lr, num_class=1, num_chn = 2, PRETRAIN=False, BG=False):
    sh  = (256,256)
    f_num= 32
    flair = Input(flair_shape, name='input_flair')
    iam = Input(iam_shape, name='input_iam')
    T1w = Input(flair_shape, name='input_T1w')
    
    conv_f1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_f_0_0')(flair)
    conv_f1 = BatchNormalization(name='bn_f_0_0')(conv_f1)
    conv_f1 = Activation('relu', name='relu_f_0_0')(conv_f1)
    conv_f1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_f_0_1')(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f_0_1')(conv_f1)
    conv_f1 = Activation('relu', name='relu_f_0_1')(conv_f1)
    pool_f = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_f_0')(conv_f1)
    
    conv_t1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_t_0_0')(T1w)
    conv_t1 = BatchNormalization(name='bn_t_0_0')(conv_t1)
    conv_t1 = Activation('relu', name='relu_t_0_0')(conv_t1)
    conv_t1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_t_0_1')(conv_t1)
    conv_t1 = BatchNormalization(name='bn_t_0_1')(conv_t1)
    conv_t1 = Activation('relu', name='relu_t_0_1')(conv_t1)
    pool_t = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_t_0')(conv_t1)
    
    conv_f = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_f_1_2')(pool_f)
    conv_f = BatchNormalization(name='bn_f_1_2')(conv_f)
    conv_f = Activation('relu', name='relu_f_1_2')(conv_f)
    conv_f = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_f_1_3')(conv_f)
    conv_f = BatchNormalization(name='bn_f_1_3')(conv_f)
    conv_f = Activation('relu', name='relu_f_1_3')(conv_f)
    
    conv_t = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_t_1_2')(pool_t)
    conv_t = BatchNormalization(name='bn_t_1_2')(conv_t)
    conv_t = Activation('relu', name='relu_t_1_2')(conv_t)
    conv_t = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_t_1_3')(conv_t)
    conv_t = BatchNormalization(name='bn_t_1_3')(conv_t)
    conv_t = Activation('relu', name='relu_t_1_3')(conv_t)
    
    conv_i1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_i_0_0')(iam)
    conv_i1 = BatchNormalization(name='bn_i_0_0')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_0')(conv_i1)
    conv_i1 = Conv2D(f_num, (11,11), padding='same', name='conv2d_i_0_1')(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i_0_1')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_1')(conv_i1)
    pool_i = MaxPooling2D(pool_size=(2,2), name='maxpool2d_i_1')(conv_i1)
    
    conv_i = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_i_1_2')(pool_i)
    conv_i = BatchNormalization(name='bn_i_1_2')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_2')(conv_i)
    conv_i = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_i_1_3')(conv_i)
    conv_i = BatchNormalization(name='bn_i_1_3')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_3')(conv_i)
 
    concat1 = concatenate([conv_f, conv_t, conv_i], axis=3, name='concat_0_0')
    pool_f = MaxPooling2D(pool_size=(2, 2), name='miaxpool2d_f_1_2')(concat1)
   
    conv_1 = Conv2D(f_num*4, (3,3), padding='same', name='conv2d_2_4')(pool_f)
    conv_1 = BatchNormalization(name='bn_2_4')(conv_1)
    conv_1 = Activation('relu', name='relu_2_4')(conv_1)
    conv_1 = Dropout(0.25, name='do_2_0')(conv_1)
    conv_1 = Conv2D(f_num*4, (3,3), padding='same', name='conv2d_2_5')(conv_1)
    conv_1 = BatchNormalization(name='bn_2_5')(conv_1)
    conv_1 = Activation('relu', name='relu_2_5')(conv_1)
    
    up1 = UpSampling2D(size=(2, 2), name="up2d_2_0")(conv_1)
    up1 = concatenate([up1, concat1], axis=3, name='concat_2_1')
    
    conv_2 = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_3_6')(up1)
    conv_2 = BatchNormalization(name='bn_3_6')(conv_2)
    conv_2 = Activation('relu', name='relu_3_6')(conv_2)
    conv_2 = Dropout(0.25, name='do_3_1')(conv_2)
    conv_2 = Conv2D(f_num*2, (7,7), padding='same', name='conv2d_3_7')(conv_2)
    conv_2 = BatchNormalization(name='bn_3_7')(conv_2)
    conv_2 = Activation('relu', name='relu_3_7')(conv_2)
    
    up2 = UpSampling2D(size=(2, 2), name="up2d_3_1")(conv_2)
    up2 = concatenate([up2, conv_f1, conv_t1, conv_i1], axis=3, name='concat_3_2')
    
    conv_2 = Conv2D(f_num, (11,11), padding='same', name='conv2d_4_8')(up2)
    conv_2 = BatchNormalization(name='bn_4_8')(conv_2)
    conv_2 = Activation('relu', name='relu_4_8')(conv_2)
    conv_2 = Dropout(0.25, name='do_4_2')(conv_2)
    conv_2 = Conv2D(f_num, (11,11), padding='same', name='conv2d_4_9')(conv_2)
    conv_2 = BatchNormalization(name='bn_4_9')(conv_2)
    conv_2 = Activation('relu', name='relu_4_9')(conv_2)
    
    
    conv_3 = Conv2D(num_class, (1, 1), padding='same', activation=activation, name='conv2d_5_10')(conv_2)

    model = Model(input=[flair, iam, T1w], output=conv_3)
    
    if PRETRAIN:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[binary_accuracy, dice_coef])

    
    return model 

def Building_UNet(layer_level, batch_norm, input_shape, Loss, activation, lr, num_class=1, PRETRAIN=False, BG=False):
    inputs = conv = Input(input_shape)
    unet_ch = [64, 96, 128, 256, 512, 1024]
    ch_stack = []
    conv_stack = []
    
    conv_idx = bn_idx = relu_idx = pool_idx = up_idx = concat_idx = do_idx = 0
        
    # Downsampling Network
    for i in range(layer_level+1):
        ch_stack.append(unet_ch[i])
        for j in range(2):
            # First layer receptive field = 5, 5 
            if i == 0:
                conv = Conv2D(unet_ch[i], (5, 5), padding='same', 
                              name='conv2d_'+str(i)+'_'+str(conv_idx))(conv)
            else:
                conv = Conv2D(unet_ch[i], (3, 3), padding='same', 
                              name='conv2d_'+str(i)+'_'+str(conv_idx))(conv)
            
            if batch_norm:
                conv = BatchNormalization(name='bn_'+str(i)+'_'+str(bn_idx))(conv)
            conv = Activation('relu', name='relu_'+str(i)+'_'+str(relu_idx))(conv)
            
            # Add Dropout
            if i == layer_level and j ==0:
                conv = Dropout(0.25, name='do_'+str(i)+'_'+str(do_idx))(conv)
                do_idx = do_idx+1
                
            conv_idx = conv_idx+1
            bn_idx = bn_idx+1
            relu_idx = relu_idx+1
        
        # Pooling layer 
        if i < layer_level:
            conv_stack.append(conv)
            conv = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+str(i)+'_'+str(pool_idx))(conv)
            pool_idx = pool_idx+1
        else:
            up = UpSampling2D(size=(2, 2), name='up2d_'+str(i)+'_'+str(up_idx))(conv)
            conv = concatenate([up, conv_stack.pop()], axis=3, name = 'concat_'+str(i)+'_'+str(concat_idx))
    
    # Upsampling Network
    ch_stack.pop()
    for i in range(layer_level+1, layer_level*2+1):
        ch_num = ch_stack.pop()
        for j in range(2):
            conv = Conv2D(ch_num, (3, 3), padding='same', 
                          name='conv2d_'+str(i)+'_'+str(conv_idx))(conv)
            
            if batch_norm:
                conv = BatchNormalization(name='bn_'+str(i)+'_'+str(bn_idx))(conv)
            conv = Activation('relu', name='relu_'+str(i)+'_'+str(relu_idx))(conv)
            conv_idx = conv_idx+1
            # Add Dropout
            if i == layer_level+1 and j == 0:
                conv = Dropout(0.25, name='do_'+str(i)+'_'+str(do_idx))(conv)
                do_idx = do_idx+1
            
            bn_idx = bn_idx+1
            relu_idx = relu_idx+1
        
        # Upsampling layer 
        if i < layer_level*2:
            up = UpSampling2D(size=(2, 2), name='up2d_'+str(i)+'_'+str(up_idx))(conv)
            conv = concatenate([up, conv_stack.pop()], axis=3, name = 'concat_'+str(i)+'_'+str(concat_idx))
        else:
            conv = Conv2D(num_class, (1, 1), padding='same', activation=activation,
                          name = 'conv2d_'+str(i+1)+'_'+str(conv_idx))(conv)
        
    model = Model(input=inputs, output=conv)
    
    if PRETRAIN:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[binary_accuracy, dice_coef])
    
    return model

def Saliency_UNet(layer_level, batch_norm, flair_shape, iam_shape, Loss, activation, lr, num_class=1, PRETRAIN=False, BG=False):
 
    flair = Input(flair_shape, name='input_flair')
    iam = Input(iam_shape, name='input_iam')
    
    conv_f1 = Conv2D(64, (3,3), padding='same', name='conv2d_f_0_0')(flair)
    conv_f1 = BatchNormalization(name='bn_f_0_0')(conv_f1)
    conv_f1 = Activation('relu', name='relu_f_0_0')(conv_f1)
    conv_f1 = Conv2D(64, (3,3), padding='same', name='conv2d_f_0_1')(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f_0_1')(conv_f1)
    conv_f1 = Activation('relu', name='relu_f_0_1')(conv_f1)
    
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i_0_0')(iam)
    conv_i1 = BatchNormalization(name='bn_i_0_0')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_0')(conv_i1)
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i_0_1')(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i_0_1')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_1')(conv_i1)
    
    pool_f = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_f_1_0')(conv_f1)
    conv_f = Conv2D(128, (3,3), padding='same', name='conv2d_f_1_2')(pool_f)
    conv_f = BatchNormalization(name='bn_f_1_2')(conv_f)
    conv_f = Activation('relu', name='relu_f_1_2')(conv_f)
    conv_f = Conv2D(128, (3,3), padding='same', name='conv2d_f_1_3')(conv_f)
    conv_f = BatchNormalization(name='bn_f_1_3')(conv_f)
    conv_f = Activation('relu', name='relu_f_1_3')(conv_f)
    
    pool_i = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_i_1_0')(conv_i1)
    conv_i = Conv2D(128, (3,3), padding='same', name='conv2d_i_1_2')(pool_i)
    conv_i = BatchNormalization(name='bn_i_1_2')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_2')(conv_i)
    conv_i = Conv2D(128, (3,3), padding='same', name='conv2d_i_1_3')(conv_i)
    conv_i = BatchNormalization(name='bn_i_1_3')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_3')(conv_i)
    
    concat1 = concatenate([conv_f, conv_i], axis=3, name='concat_1_0')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1_1')(concat1)
    
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_4')(pool1)
    conv_1 = BatchNormalization(name='bn_2_4')(conv_1)
    conv_1 = Activation('relu', name='relu_2_4')(conv_1)
    conv_1 = Dropout(0.25, name='do_2_0')(conv_1)
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_5')(conv_1)
    conv_1 = BatchNormalization(name='bn_2_5')(conv_1)
    conv_1 = Activation('relu', name='relu_2_5')(conv_1)
    
    up1 = UpSampling2D(size=(2, 2), name="up2d_2_0")(conv_1)
    up1 = concatenate([up1, concat1], axis=3, name='concat_2_1')
    
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_6')(up1)
    conv_2 = BatchNormalization(name='bn_3_6')(conv_2)
    conv_2 = Activation('relu', name='relu_3_6')(conv_2)
    conv_2 = Dropout(0.25, name='do_3_1')(conv_2)
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_7')(conv_2)
    conv_2 = BatchNormalization(name='bn_3_7')(conv_2)
    conv_2 = Activation('relu', name='relu_3_7')(conv_2)
    
    up2 = UpSampling2D(size=(2, 2), name="up2d_3_1")(conv_2)
    up2 = concatenate([up2, conv_f1, conv_i1], axis=3, name='concat_3_2')
        
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_8')(up2)
    conv_2 = BatchNormalization(name='bn_4_8')(conv_2)
    conv_2 = Activation('relu', name='relu_4_8')(conv_2)
    conv_2 = Dropout(0.25, name='do_4_2')(conv_2)
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_9')(conv_2)
    conv_2 = BatchNormalization(name='bn_4_9')(conv_2)
    conv_2 = Activation('relu', name='relu_4_9')(conv_2)
    
    
    conv_3 = Conv2D(num_class, (1, 1), padding='same', activation=activation, name='conv2d_5_10')(conv_2)

    model = Model(input=[flair, iam], output=conv_3)
    
    if PRETRAIN:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[binary_accuracy, dice_coef])
    
    return model

def Saliency_Encoder(input_ch, name, activation):
    activation_layer = activation.layer
    activation_name = activation.name+'_'+name
    
    conv_1 = Conv2D(64, (5,5), padding='same', name='conv2d_'+name+'_0_0')(input_ch)
    conv_1 = BatchNormalization(name='bn_'+name+'_0_0')(conv_1)
    conv_1 = activation_layer(name=activation_name+'_0_0')(conv_1)
    conv_1 = Conv2D(64, (5,5), padding='same', name='conv2d_'+name+'_0_1')(conv_1)
    conv_1 = BatchNormalization(name='bn_'+name+'_0_1')(conv_1)
    conv_1 = activation_layer(name=activation_name+'_0_1')(conv_1)
    if name == 't':
        pool_1 = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_'+name+'_0_0')(conv_1)
    else:
        pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+name+'_0_0')(conv_1)

    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_'+name+'_1_2')(pool_2)
    conv_2 = BatchNormalization(name='bn_'+name+'_1_2')(conv_2)
    conv_2 = activation_layer(name=activation_name+'_1_2')(conv_2)
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_'+name+'_1_3')(conv_2)
    conv_2 = BatchNormalization(name='bn_'+name+'_1_3')(conv_2)
    conv_2 = activation_layer(name=activation_name+'_1_3')(conv_2)
    if name == 't':
        pool_2 = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_'+name+'_1_1')(conv_2)
    else:
        pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+name+'_1_1')(conv_2)
    
    return conv_1, conv_2, pool_2
    
class act_layer:
    def __init__(self,activation, name):
        self.layer = activation
        self.name = name    

def Saliency_UNet(input_shape, Loss, activation, lr,num_ch, num_class=1):
    
    flair = Input(input_shape, name='input_flair')
    iam = Input(input_shape, name='input_iam')
    input_ch = [flair, iam]

    
    conv_f1, conv_f, pool_f = Saliency_Encoder(flair, 'f', act_layer(ReLU,'relu'))
    conv_i1, conv_i, pool_i = Saliency_Encoder(flair, 'i', act_layer(ReLU,'relu'))
    if num_ch == 3:
        T1w = Input(input_shape, name='input_T1w')
        input_ch.append(T1w)
        conv_t1, conv_t, pool_t = Saliency_Encoder(T1w, 't', act_layer(LeakyReLU,'Lrelu'))
        conocat_ch0 = [pool_f, pool_i, pool_t]
        concat_ch1 = [conv_f1, conv_i1, conv_t1]
        concat_ch2 = [conv_f, conv_i, conv_t]
    else:
        concat_ch0 = [pool_f, pool_i]
        concat_ch1 = [conv_f1, conv_i1]
        concat_ch2 = [conv_f, conv_i]
    
    concat1 = concatenate(concat_ch0, axis=-1, name='concat_0_0')

    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_4')(concat1)
    conv_1 = BatchNormalization(name='bn_2_4')(conv_1)
    conv_1 = Activation('relu', name='relu_2_4')(conv_1)
    conv_1 = Dropout(0.25, name='do_2_0')(conv_1)
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_5')(conv_1)
    conv_1 = BatchNormalization(name='bn_2_5')(conv_1)
    conv_1 = Activation('relu', name='relu_2_5')(conv_1)
    
    up1 = UpSampling2D(size=(2, 2), name="up2d_2_0")(conv_1)
    up1 = concatenate([up1]+concat_ch2, axis=-1, name='concat_2_1')
    
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_6')(up1)
    conv_2 = BatchNormalization(name='bn_3_6')(conv_2)
    conv_2 = Activation('relu', name='relu_3_6')(conv_2)
    conv_2 = Dropout(0.25, name='do_3_1')(conv_2)
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_7')(conv_2)
    conv_2 = BatchNormalization(name='bn_3_7')(conv_2)
    conv_2 = Activation('relu', name='relu_3_7')(conv_2)
    
    up2 = UpSampling2D(size=(2, 2), name="up2d_3_1")(conv_2)
    up2 = concatenate([up2]+concat_ch1, axis=-1, name='concat_3_2')
        
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_8')(up2)
    conv_2 = BatchNormalization(name='bn_4_8')(conv_2)
    conv_2 = Activation('relu', name='relu_4_8')(conv_2)
    conv_2 = Dropout(0.25, name='do_4_2')(conv_2)
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_9')(conv_2)
    conv_2 = BatchNormalization(name='bn_4_9')(conv_2)
    conv_2 = Activation('relu', name='relu_4_9')(conv_2)
    
    conv_3 = Conv2D(num_class, (1, 1), padding='same', activation=activation, name='conv2d_5_10')(conv_2)

    model = Model(input=input_ch, output=conv_3)
    
    model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
     
    return model

def Saliency_UNet_v3(layer_level, batch_norm, flair_shape, iam_shape, Loss, activation, lr, num_class=1, num_chn = 2, PRETRAIN=False, BG=False):
    
    def weight_learning_block(conv_f,conv_t,conv_i, idx):
        idx=str(idx)
        ## Weight Learning
        concat_lw = concatenate([conv_f, conv_t, conv_i], axis=3, name='concat_lw'+idx)
        conv_lw = Conv2D(256, (3,3), padding='same', name='conv2d_lw'+idx+'_0')(concat_lw)
        conv_lw = Activation('relu', name='relu_lw'+idx)(conv_lw)
        conv_lw = Conv2D(3, (1,1), padding='same', name='conv2d_lw'+idx+'_1', activation='softmax')(conv_lw)
        
        ## Multiply weights with each channel   
        outs = Lambda(lambda x: tf.stack(x,-1), name='stack_'+idx)([conv_f, conv_t, conv_i])
        at_map = Lambda(lambda x: tf.expand_dims(x, axis=-2),name='weight_map_'+idx)(conv_lw)
        mul_map =Lambda(lambda x: tf.multiply(x[0],x[1]), name='mul_'+idx)([at_map, outs])
        concat1 = Lambda(lambda x: tf.unstack(x, axis=-1), name='unstack_'+idx)(mul_map)
        
        return concat1
    
    def multimodal_maxpooling(conv_f, conv_t, conv_i):
        conv_f = Lambda(lambda x: tf.expand_dims(x, axis=-2),name='expand_dims_f')(conv_f)
        conv_t = Lambda(lambda x: tf.expand_dims(x, axis=-2),name='expand_dims_t')(conv_t)
        conv_i = Lambda(lambda x: tf.expand_dims(x, axis=-2),name='expand_dims_i')(conv_i)
        
        concat = concatenate([conv_f,conv_t,conv_i], axis=-2, name='concat_0_2')
        
        pool = MaxPooling3D(pool_size=(2, 2, 3), name='maxpool3d_0_0')(concat)
        
        lambda_l= Lambda(lambda x: tf.squeeze(x, axis=3),name='squeeze')(pool)
        return lambda_l
        
    sh  = (256,256)
    shape = []
    flair = Input(flair_shape, name='input_flair')
    iam = Input(iam_shape, name='input_iam')
    T1w = Input(flair_shape, name='input_T1w')
    
    conv_f1 = Conv2D(64, (5,5), padding='same', name='conv2d_f_0_0')(flair)
    conv_f1 = BatchNormalization(name='bn_f_0_0')(conv_f1)
    conv_f1 = LeakyReLU(alpha=0.5, name='relu_f_0_0')(conv_f1)
    conv_f1 = Conv2D(64, (5,5), padding='same', name='conv2d_f_0_1')(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f_0_1')(conv_f1)
    conv_f1 = LeakyReLU(alpha=0.5, name='relu_f_0_1')(conv_f1)
    pool_f1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_f_0')(conv_f1)
    
    conv_t1 = Conv2D(64, (5,5), padding='same', name='conv2d_t_0_0')(T1w)
    conv_t1 = BatchNormalization(name='bn_t_0_0')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t_0_0')(conv_t1)
    conv_t1 = Conv2D(64, (5,5), padding='same', name='conv2d_t_0_1')(conv_t1)
    conv_t1 = BatchNormalization(name='bn_t_0_1')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t_0_1')(conv_t1)
    pool_t1 = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_t_0')(conv_t1)
    
    conv_f = Conv2D(128, (5,5), padding='same', name='conv2d_f_1_2')(pool_f1)
    conv_f = BatchNormalization(name='bn_f_1_2')(conv_f)
    conv_f = LeakyReLU(alpha=0.5, name='relu_f_1_2')(conv_f)
    conv_f = Conv2D(128, (5,5), padding='same', name='conv2d_f_1_3')(conv_f)
    conv_f = BatchNormalization(name='bn_f_1_3')(conv_f)
    conv_f = LeakyReLU(alpha=0.5, name='relu_f_1_3')(conv_f)
    pool_f = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_f_1_1')(conv_f)
    
    conv_t = Conv2D(128, (3,3), padding='same', name='conv2d_t_1_2')(pool_t1)
    conv_t = BatchNormalization(name='bn_t_1_2')(conv_t)
    conv_t = LeakyReLU(alpha=0.5,name='Lrelu_t_1_2')(conv_t)
    conv_t = Conv2D(128, (3,3), padding='same', name='conv2d_t_1_3')(conv_t)
    conv_t = BatchNormalization(name='bn_t_1_3')(conv_t)
    conv_t = LeakyReLU(alpha=0.5,name='Lrelu_t_1_3')(conv_t)
    pool_t = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_t_1')(conv_t)
    
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i_0_0')(iam)
    conv_i1 = BatchNormalization(name='bn_i_0_0')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_0')(conv_i1)
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i_0_1')(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i_0_1')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i_0_1')(conv_i1)
    pool_i1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_i_0')(conv_i1)

    conv_i = Conv2D(128, (3,3), padding='same', name='conv2d_i_1_2')(pool_i1)
    conv_i = BatchNormalization(name='bn_i_1_2')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_2')(conv_i)
    conv_i = Conv2D(128, (3,3), padding='same', name='conv2d_i_1_3')(conv_i)
    conv_i = BatchNormalization(name='bn_i_1_3')(conv_i)
    conv_i = Activation('relu', name='relu_i_1_3')(conv_i)
    pool_i = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_i_1_1')(conv_i)

    #concat1=weight_learning_block(pool_f,pool_t,pool_i, 0)
    concat1 = concatenate([pool_f,pool_t,pool_i], axis=-1, name='concat_0_0')
    concat2 = concatenate([conv_f1, conv_t1, conv_i1], axis=-1, name='concat_0_1')

    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_4')(concat1)
    conv_1 = BatchNormalization(name='bn_2_4')(conv_1)
    conv_1 = Activation('relu', name='relu_2_4')(conv_1)
    conv_1 = Dropout(0.25, name='do_2_0')(conv_1)
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_5')(conv_1)
    conv_1 = BatchNormalization(name='bn_2_5')(conv_1)
    conv_1 = Activation('relu', name='relu_2_5')(conv_1)
    
    up1 = UpSampling2D(size=(2, 2), name="up2d_2_1")(conv_1)
    up1 = concatenate([up1, conv_f,conv_t,conv_i], axis=3, name='concat_2_2')
    
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_6')(up1)
    conv_2 = BatchNormalization(name='bn_3_6')(conv_2)
    conv_2 = Activation('relu', name='relu_3_6')(conv_2)
    conv_2 = Dropout(0.25, name='do_3_1')(conv_2)
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_7')(conv_2)
    conv_2 = BatchNormalization(name='bn_3_7')(conv_2)
    conv_2 = Activation('relu', name='relu_3_7')(conv_2)
    
    up2 = UpSampling2D(size=(2, 2), name="up2d_3_1")(conv_2)
    up2 = concatenate([up2, concat2], axis=3, name='concat_3_3')
    
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_8')(up2)
    conv_2 = BatchNormalization(name='bn_4_8')(conv_2)
    conv_2 = Activation('relu', name='relu_4_8')(conv_2)
    conv_2 = Dropout(0.25, name='do_4_2')(conv_2)
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_9')(conv_2)
    conv_2 = BatchNormalization(name='bn_4_9')(conv_2)
    conv_2 = Activation('relu', name='relu_4_9')(conv_2)
    
    
    conv_3 = Conv2D(num_class, (1, 1), padding='same', activation=activation, name='conv2d_5_10')(conv_2)

    model = Model(input=[flair, iam, T1w], output=conv_3)
    
    if PRETRAIN:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[binary_accuracy, dice_coef])

    
    return model

def Dilated_Saliency_UNet(layer_level, batch_norm, flair_shape, iam_shape, Loss, activation, lr, num_class=1, num_chn = 2, PRETRAIN=False, BG=False):
    
  
        
    sh  = (256,256)
    shape = []
    flair = Input(flair_shape, name='input_flair')
    iam = Input(iam_shape, name='input_iam')
    T1w = Input(flair_shape, name='input_T1w')
    
    # Global Context Learning - Dilated Convolution
    
    conv_f1 = Conv2D(64, (3,3), padding='same', name='conv2d_f1_0_0', dilation_rate = 1)(flair)
    conv_f1 = BatchNormalization(name='bn_f1_0_0')(conv_f1)
    conv_f1 = Activation('relu',name='relu_f1_0_0')(conv_f1)
    conv_f1 = Conv2D(64, (3,3), padding='same', name='conv2d_f1_0_1', dilation_rate = 2)(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f1_0_1')(conv_f1)
    conv_f1 = Activation('relu',name='relu_f1_0_1')(conv_f1)
    conv_f1 = Conv2D(128, (3,3), padding='same', name='conv2d_f1_0_2', dilation_rate = 4)(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f1_0_2')(conv_f1)
    conv_f1 = Activation('relu',name='relu_f1_0_2')(conv_f1)
    conv_f1 = Conv2D(128, (3,3), padding='same', name='conv2d_f1_0_3', dilation_rate = 2)(conv_f1)
    conv_f1 = BatchNormalization(name='bn_f1_0_3')(conv_f1)
    conv_f1 = Activation('relu',name='relu_f1_0_3')(conv_f1)
    pool_f1 = MaxPooling2D(pool_size=(2, 2), name='avgpool2d_f1_0')(conv_f1)
    
    conv_t1 = Conv2D(64, (3,3), padding='same', name='conv2d_t1_0_0', dilation_rate = 1)(T1w)
    conv_t1 = BatchNormalization(name='bn_t1_0_0')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t1_0_0')(conv_t1)
    conv_t1 = Conv2D(64, (3,3), padding='same', name='conv2d_t1_0_1', dilation_rate = 2)(conv_t1)
    conv_t1 = BatchNormalization(name='bn_t1_0_1')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t1_0_1')(conv_t1)
    conv_t1 = Conv2D(128, (3,3), padding='same', name='conv2d_t1_0_2', dilation_rate = 4)(conv_t1)
    conv_t1 = BatchNormalization(name='bn_t1_0_2')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t1_0_2')(conv_t1)
    conv_t1 = Conv2D(128, (3,3), padding='same', name='conv2d_t1_0_3', dilation_rate = 2)(conv_t1)
    conv_t1 = BatchNormalization(name='bn_t1_0_3')(conv_t1)
    conv_t1 = LeakyReLU(alpha=0.5,name='Lrelu_t1_0_3')(conv_t1)
    pool_t1 = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_t1_0')(conv_t1)
    
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i1_0_0', dilation_rate = 1)(iam)
    conv_i1 = BatchNormalization(name='bn_i1_0_0')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i1_0_0')(conv_i1)
    conv_i1 = Conv2D(64, (3,3), padding='same', name='conv2d_i1_0_1', dilation_rate = 2)(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i1_0_1')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i1_0_1')(conv_i1)
    conv_i1 = Conv2D(128, (3,3), padding='same', name='conv2d_i1_0_2', dilation_rate = 4)(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i1_0_2')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i1_0_2')(conv_i1)
    conv_i1 = Conv2D(128, (3,3), padding='same', name='conv2d_i1_0_3', dilation_rate = 2)(conv_i1)
    conv_i1 = BatchNormalization(name='bn_i1_0_3')(conv_i1)
    conv_i1 = Activation('relu', name='relu_i1_0_3')(conv_i1)
    pool_i1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_i1_0')(conv_i1)

    conv_f = Conv2D(128, (3,3), padding='same', name='conv2d_f_4')(pool_f1)
    conv_f = BatchNormalization(name='bn_f_4')(conv_f)
    conv_f = Activation('relu',name='relu_f_4')(conv_f)
   
    conv_t = Conv2D(128, (3,3), padding='same', name='conv2d_t_4')(pool_t1)
    conv_t = BatchNormalization(name='bn_t_4')(conv_t)
    conv_t = LeakyReLU(alpha=0.5, name='Lrelu_t_4')(conv_t)
    
    conv_i = Conv2D(128, (3,3), padding='same', name='conv2d_i_4')(pool_i1)
    conv_i = BatchNormalization(name='bn_i_4')(conv_i)
    conv_i = Activation('relu', name='relu_i_4')(conv_i)
    
    concat1 = concatenate([conv_f,conv_t,conv_i], axis=-1, name='concat_0_1')

    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_i_2_0')(concat1)
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_4')(pool_1)
    conv_1 = BatchNormalization(name='bn_2_4')(conv_1)
    conv_1 = Activation('relu', name='relu_2_4')(conv_1)
    conv_1 = Dropout(0.25, name='do_2_0')(conv_1)
    conv_1 = Conv2D(256, (3,3), padding='same', name='conv2d_2_5')(conv_1)
    conv_1 = BatchNormalization(name='bn_2_5')(conv_1)
    conv_1 = Activation('relu', name='relu_2_5')(conv_1)
    
    up1 = UpSampling2D(size=(2, 2), name="up2d_2_1")(conv_1)
    up1 = concatenate([up1, concat1], axis=3, name='concat_2_2')
    
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_6')(up1)
    conv_2 = BatchNormalization(name='bn_3_6')(conv_2)
    conv_2 = Activation('relu', name='relu_3_6')(conv_2)
    conv_2 = Dropout(0.25, name='do_3_1')(conv_2)
    conv_2 = Conv2D(128, (3,3), padding='same', name='conv2d_3_7')(conv_2)
    conv_2 = BatchNormalization(name='bn_3_7')(conv_2)
    conv_2 = Activation('relu', name='relu_3_7')(conv_2)
    
    up2 = UpSampling2D(size=(2, 2), name="up2d_3_1")(conv_2)
    up2 = concatenate([up2, conv_f1, conv_t1, conv_i1], axis=3, name='concat_3_3')
    
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_8')(up2)
    conv_2 = BatchNormalization(name='bn_4_8')(conv_2)
    conv_2 = Activation('relu', name='relu_4_8')(conv_2)
    conv_2 = Dropout(0.25, name='do_4_2')(conv_2)
    conv_2 = Conv2D(64, (3,3), padding='same', name='conv2d_4_9')(conv_2)
    conv_2 = BatchNormalization(name='bn_4_9')(conv_2)
    conv_2 = Activation('relu', name='relu_4_9')(conv_2)
    
    
    conv_3 = Conv2D(num_class, (1, 1), padding='same', activation=activation, name='conv2d_5_10')(conv_2)

    model = Model(input=[flair, iam, T1w], output=conv_3)
    
    if PRETRAIN:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
    else:
        model.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[binary_accuracy, dice_coef])

    
    return model

