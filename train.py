import os, time, h5py, argparse, sys

import scipy.io as sio
import numpy as np
import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as K

from timeit import default_timer as timer

from utils import TestCallback
from model import Net_Model


def train_model(config, START_TIME, net_depth, SALIENCY, DILATION, restore_dir, net_name, train_dat, test_dat):
    
    train_data = train_dat[0]
    train_trgt = train_dat[1]
    test_data = test_dat[0]
    test_trgt = test_dat[1]
    num_chn = np.shape(train_dat)[-1]
    
    elapsed_times_all = np.zeros((config.fold))
    
    for b_id in range(fold):  
        '''Building Network'''
        my_network = Net_Model(net_depth, (None, None, num_chn), 
                                Loss=config.loss, 
                                lr=config.lr, 
                                num_chn = num_chn,
                                num_class=config.n_class,
                                VISUALISATION = config.VISUALISATION,
                                PATCH = config.Patch,
                                SALIENCY=SALIENCY,
                                DILATION=DILATION)
        
        net_name = net_type+'_'+START_TIME+'_'+my_network.net_name
        print("\nBuilt network: "+my_network.net_name+"...")
        
        if not restore_dir:
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
        
        # History Callbacks
        tensorboard = TensorBoard(log_dir = saving_dir + 'tensorboard_log/', batch_size = nb_samples, histogram_freq = hist_freq) 
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=args.reduce_lr_factor, mode='max', patience=args.reduce_lr_patience, min_lr=2e-10)
        test_callback = TestCallback(test_data, test_trgt)
        
        # Train network
        my_network.train(train_data, train_trgt, 0.2, num_epochs, nb_samples, [reduce_lr, test_callback, tensorboard])
        
        ## Save Results
        elapsed_times_all[b_id] = timer() - one_timer
        my_network.save_statistic_results(saving_dir, elapsed_times_all)
        my_network.save_img_results(saving_dir, test_data, test_trgt)
        
