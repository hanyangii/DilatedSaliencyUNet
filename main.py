import os, time, h5py, argparse, sys

import scipy.io as sio
import numpy as np
import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from utils import set_parser, TrainConfig
from data import generate_patch_data, generate_slice_data
from train import train_model

START_TIME = time.strftime('%Y%m%d-%H%M', time.gmtime())

'''Handling Parameter'''
parser = argparse.ArgumentParser()
parser = set_parser(parser)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
np.random.seed(42)
tf.set_random_seed(42)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

if __name__ == '__main__':
    
    #print_options(vars(args))
    
    # Parameter Setting
    TRSH          = args.TRSH
    random_num    = args.random_num
    RESTORE       = args.restore
    data_chn_num  = args.data_chn_num
    
    if not args.Patch:
        img_x, img_y, img_z = 256, 256, 1
    else:
        img_x, img_y, img_z = args.img_size, args.img_size, 1

    #patch_size = str(img_x)+'size_chn'+str(data_chn_num)
    #net_name = args.dir_name+'_'+patch_size
    
    ''' Load Data '''
    train_config_dir = 'data/com_test_configs_2fold_adni60'
    test_config_dir = 'data/com_test_configs_2fold_adni60'
    
    restore_weights_path = 'results/basic_experiments_9/'
    
    win_shape = (img_x, img_y, img_z)

    print("Reading TRAIN data..")
    train_data, train_trgt, train_list = generate_patch_data(train_config_dir, 0, TRSH, win_shape, random_num, data_chn_num, args.test)    
    
    print("Reading TEST data..")
    test_data, test_trgt, target_list = generate_slice_data(test_config_dir, 1, random_num, data_chn_num, args.test)
      
    print(train_data.shape, ' class max val:',np.max(train_trgt))

    # Reshape datat to 4-dims
    # Data Order [FLAIR, IAM, T1W]
    train_data = [np.expand_dims(train_data[:,:,:,i], axis=3) for i in range(data_chn_num)]
    test_data = [np.expand_dims(test_data[:,:,:,i], axis=3) for i in range(data_chn_num)]
    
    
    ''' Train Networks'''
    train_config = TrainConfig(args)
    '''
    # U-Net (only FLAIR)
    
    train_dat = [train_data[0], train_trgt]
    test_dat = [test_data[0], test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=None, net_type='UNet_depth3_FLAIR', train_dat=train_dat, test_dat=test_dat)
    
    # U-Net (only IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = [train_data[1], train_trgt]
    test_dat = [test_data[1], test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=False, DILATION=False, restore_dir=None, net_type='IAM', train_dat=train_dat, test_dat=test_dat)
    
    # U-Net (FLAIR + IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = np.concatenate(train_data[0:2], axis=3)
    test_dat = np.concatenate(test_data[0:2], axis=3)    
    train_dat = [train_dat, train_trgt]
    test_dat = [test_dat, test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=None, net_type='F+I', train_dat=train_dat, test_dat=test_dat)
    
    # U-Net (FLAIR + IAM + T1w)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = np.concatenate(train_data, axis=-1)
    test_dat = np.concatenate(test_data, axis=-1)  
    train_dat = [train_dat, train_trgt]
    test_dat = [test_dat, test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=None, net_type='All', train_dat=train_dat, test_dat=test_dat)

    
    # Saliency U-Net (FLAIR+IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = [train_data[0:2], train_trgt]
    test_dat = [test_data[0:2], test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=True, DILATION=False, restore_dir=None, net_type='F+I', train_dat=train_dat, test_dat=test_dat)


   '''
    # Dilated Saliency U-Net (FLAIR + IAM)
    #restore_weights_path+'F+I_20181220-1123_Dilated_Saliency_UNet_ep80_0/train_models.h5'
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = [train_data[0:2], train_trgt]
    test_dat = [test_data[0:2], test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=True, DILATION=True, restore_dir=None, net_type='F+I', train_dat=train_dat, test_dat=test_dat)
    '''    
    # Saliency U-Net (FLAIR+IAM+T1w)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = [train_data, train_trgt]
    test_dat = [test_data, test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=True, DILATION=False, 
                restore_dir=None, net_type='All', train_dat=train_dat, test_dat=test_dat)
    
    
    # Dilated Saliency U-Net (FLAIR + IAM + T1w)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    train_dat = [train_data, train_trgt]
    test_dat = [test_data, test_trgt]
    train_model(train_config,START_TIME, net_depth=3, SALIENCY=True, DILATION=True, 
                restore_dir=None, net_type='All', train_dat=train_dat, test_dat=test_dat)
    
    '''
    # Clear memory
    train_trgt = None
    test_trgt  = None
    train_dat = None
    targt_dat = None
    train_data = None
    test_data = None
