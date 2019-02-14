import os, time, h5py, argparse, sys

import scipy.io as sio
import numpy as np
import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from utils import set_parser, TrainConfig
from data import generate_patch_data, generate_slice_data
from train import test_model

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
    test_config_dir = 'data/com_test_configs_2fold_adni60'
    
    restore_weights_path = 'results/basic_experiments_9/'
    
    win_shape = (img_x, img_y, img_z)

    print("Reading TEST data..")
    test_data, test_trgt, label_list, affine_list = generate_slice_data(test_config_dir, 0, random_num, data_chn_num, args.test)
      
    # Reshape datat to 4-dims
    # Data Order [FLAIR, IAM, T1W]
    test_data = [np.expand_dims(test_data[:,:,:,i], axis=3) for i in range(data_chn_num)]
    
    ''' Train Networks'''
    train_config = TrainConfig(args)
    
    # U-Net (only FLAIR)
    test_model(train_config,test_data[0], test_trgt, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=restore_weights_path+'UNet_depth3_FLAIR_20190116-0142_UNet_depth3_new_basic_ep80_0/train_models.h5', net_type='UNet_depth3_FLAIR',label_list=label_list, affine_list = affine_list)
    
    # U-Net (only IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    test_model(train_config, test_data[1], test_trgt, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=restore_weights_path+'IAM_20190116-0142_UNet_depth3_new_basic_ep80_0/train_models.h5', net_type='IAM',label_list=label_list, affine_list = affine_list)
    
    # U-Net (FLAIR + IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    test_dat = np.concatenate(test_data[0:2], axis=3)   
    test_model(train_config,test_dat, test_trgt, net_depth=3, SALIENCY=False, DILATION=False, 
                restore_dir=restore_weights_path+'F+I_20190116-0254_UNet_depth3_new_basic_ep80_0/train_models.h5', net_type='F+I', label_list=label_list, affine_list = affine_list)
    
    # Saliency U-Net (FLAIR+IAM)
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    test_model(train_config,test_data[0:2], test_trgt, net_depth=args.depth, SALIENCY=True, DILATION=False, 
                restore_dir='results/F+I_depth1_20190212-0725_Saliency_UNet_ep2_0/train_models.h5', net_type='F+I', label_list=label_list, affine_list = affine_list)
    
    # Dilated Saliency U-Net (FLAIR+IAM)
    # Dilation factor - 1242
    K.clear_session()
    sess = tf.Session(config=config)
    K.set_session(sess)
    test_model(train_config, test_data[0:2], test_trgt, net_depth=2, SALIENCY=True, DILATION=True, 
                restore_dir=restore_weights_path+'F+I_1242_20190117-0109_Dilated_Saliency_UNet_ep80_0/train_models.h5', net_type='F+I', label_list=label_list, affine_list = affine_list, dilation_factor = [[1,2],[4,2]])
    
    
    # Clear memory
    train_trgt = None
    test_trgt  = None
    train_dat = None
    targt_dat = None
    train_data = None
    test_data = None
