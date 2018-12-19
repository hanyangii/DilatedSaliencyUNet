import os, time, h5py, argparse, sys

import scipy.io as sio
import numpy as np
import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K

from timeit import default_timer as timer

from utils import TestCallback
from model import Net_Model
from UNet_lib import LossHistory


def train_model(config, START_TIME, net_depth, SALIENCY, DILATION, restore_dir, net_type, train_dat, test_dat):
    
    train_data = train_dat[0]
    train_trgt = train_dat[1]
    test_data = test_dat[0]
    test_trgt = test_dat[1]
    if isinstance(train_data, list): 
        num_chn = np.shape(train_data)[-1]
        num_modal = len(train_data)
    else: 
        num_chn = train_data.shape[-1]
        num_modal = num_chn
    
    elapsed_times_all = np.zeros((config.fold))
    
    for b_id in range(config.fold):  
        '''Building Network'''
        my_network = Net_Model(net_depth, (None, None, num_chn), 
                                Loss=config.loss, 
                                lr=config.lr, 
                                num_modal = num_modal,
                                num_chn = num_chn,
                                num_class=config.n_class,
                                VISUALISATION = config.VISUALISATION,
                                SALIENCY=SALIENCY,
                                DILATION=DILATION)
        
        net_name = net_type+'_'+START_TIME+'_'+my_network.net_name
        print("\nBuilt network: "+my_network.net_name+"...")
        
        if restore_dir:
            my_network.restore(restore_dir)
        else:
            my_network.initialize()
        
        history_batch = LossHistory()
        my_network.summary()
        one_timer = timer()
        
        # Save results 
        saving_filename = str(net_name)+'_ep'+str(config.num_epochs)+'_'+str(b_id)
        print('\n\nSaving_filename: ' + saving_filename)
        saving_dir = './results/'+saving_filename+'/'
        if os.path.exists(saving_dir):
            os.rmdir(saving_dir)
        os.mkdir(saving_dir)
        
        # to_categorical for labels
        if my_network.activation == 'softmax':
            train_trgt = to_categorical(train_trgt, num_classes = my_network.num_class)
            test_trgt = to_categorical(test_trgt, num_classes = my_network.num_class)
                
        print('\nTRAIN DATASET PERMUTED size: ',np.shape(train_data))
        print('\nTRAIN LABEL DATASET PERMUTED size: ',np.shape(train_trgt))
        print('\nTEST DATASET PERMUTED size: ',np.shape(test_data))
        print('\nTEST LABEL DATASET PERMUTED size: ' ,np.shape(test_trgt))
        
        # History Callbacks
        tensorboard = TensorBoard(log_dir = saving_dir + 'tensorboard_log/', batch_size = config.batch_size, histogram_freq = config.hist_freq) 
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=config.reduce_lr_factor, mode='max', patience=config.reduce_lr_patience, min_lr=2e-10)
        test_callback = TestCallback(test_data, test_trgt, saving_dir)
        
       
        # Train network
        my_network.train(train_data, train_trgt, 0.2, config.num_epochs, config.batch_size, [reduce_lr,  test_callback, tensorboard])
        ## Save Results
        elapsed_times_all[b_id] = timer() - one_timer
        my_network.save_statistic_results(saving_dir, elapsed_times_all)
        #my_network.save_img_results(saving_dir, test_data, test_trgt)
        
        my_netowrk = None
