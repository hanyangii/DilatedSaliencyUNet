import os, time, h5py, argparse, sys, shutil

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


def test_model(config, test_data, test_trgt,net_depth, SALIENCY, DILATION, restore_dir, net_type, label_list):
    if isinstance(test_data, list): 
        num_chn = np.shape(test_data)[-1]
        num_modal = len(test_data)
    else: 
        num_chn = test_data.shape[-1]
        num_modal = num_chn
        
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
    
    '''Restore Weights'''
    my_network.restore(restore_dir)
    save_dir = restore_dir.split('train_models.h5')[0]
    os.mkdir(save_dir+'/predictions')
    
    test_trgt = to_categorical(test_trgt, num_classes = my_network.num_class)
    
    '''Test each volume'''
    
    with open(save_dir+'/predictions/test_slice_dsc.txt', 'w') as dsc_file:
        for i, subject in enumerate(label_list):
            prediction = []
            subject = suject.split('.')[0]
            print('Predict : ', subject)
            for j in range(1):
                trgt = test_trgt[i*35+7*j:i*35+7*(j+1),:,:,:]
                if isinstance(test_data, list):
                    test_img = [modal_data[i*35+7*j:i*35+7*(j+1),:,:,:] for modal_data in test_data]
                else: 
                    test_img = test_data[i*35+7*j:i*35+7*(j+1),:,:,:]

                # Predict
                print('test data: ', np.shape(test_img))
                slice_dsc, pred_img = my_network.test_result_store(test_img, trgt)
                prediction.append(pred_img)
                for dsc in slice_dsc:
                    dsc = dsc[1:-1]+'\n'
                    dsc_file.write(str(dsc))
            
            prediction = np.concatenate(prediction, axis=0)
            prediction = np.moveaxis(prediction, 0, -1)
            print(prediction.shape)
            prediction = nib.NiftiImage(prediction, np.eye(4))
            nib.save(prediction, save_dir+'predictions/'+subject+'.nii.gz')
        

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
    
    print('num chn: ', num_chn, ' / num_modal: ', num_modal)
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
        if len(config.dir_name)>0: net_name = net_name+'_'+config.dir_name
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
            shutil.rmtree(saving_dir)
        os.mkdir(saving_dir)
        
        # Validation Data
        split_idx = int(np.shape(train_trgt)[0]*0.8)
        valid_data = []
        print(split_idx)
        if len(np.shape(train_data)) == 4:
            valid_data = train_data[split_idx:-1 , :, :, :]
            train_data = train_data[0:split_idx, :, :, :]
        else:
            new_train_data = []
            for modal in train_data:
                valid_data.append(modal[split_idx:-1 , :, :, :])
                new_train_data.append(modal[0:split_idx, :, :, :])
            train_data = new_train_data
            new_train_data = None
        valid_trgt = train_trgt[split_idx:-1 , :, :, :]
        train_trgt = train_trgt[0:split_idx, :, :, :]
        
        # to_categorical for labels
        if my_network.activation == 'softmax':
            train_trgt = to_categorical(train_trgt, num_classes = my_network.num_class)
            original_test_trgt = test_trgt
            test_trgt = to_categorical(test_trgt, num_classes = my_network.num_class)
            original_valid_trgt = valid_trgt
            valid_trgt = to_categorical(valid_trgt, num_classes = my_network.num_class)
            
                
        print('\nTRAIN DATASET PERMUTED size: ',np.shape(train_data))
        print('\nTRAIN LABEL DATASET PERMUTED size: ',np.shape(train_trgt))
        print('\VALID DATASET PERMUTED size: ',np.shape(valid_data))
        print('\VALID LABEL DATASET PERMUTED size: ',np.shape(valid_trgt))
        print('\nTEST DATASET PERMUTED size: ',np.shape(test_data))
        print('\nTEST LABEL DATASET PERMUTED size: ' ,np.shape(test_trgt))
        
        
        
        # History Callbacks
        tensorboard = TensorBoard(log_dir = saving_dir + 'tensorboard_log/', batch_size = config.batch_size, histogram_freq = config.hist_freq) 
        reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=config.reduce_lr_factor, mode='max', patience=config.reduce_lr_patience, min_lr=2e-10)
        test_callback = TestCallback(test_data, test_trgt, saving_dir)
        
       
        # Train network
        if config.interim_vis:
            os.mkdir(saving_dir+'interim_results/')
            epochs = 0
            epochs_num = int(config.num_epochs/5.0)
            for i in range(epochs_num):
                epochs = int(epochs+5)
                my_network.train(train_data, train_trgt, valid_data, valid_trgt, epochs, config.batch_size, [reduce_lr,  test_callback, tensorboard])
                my_network.save_img_results(saving_dir+'interim_results/', test_data, original_test_trgt, epochs)
                my_network.save_img_results(saving_dir+'interim_results/val_', valid_data, original_valid_trgt, epochs)

        else:        
            my_network.train(train_data, train_trgt, valid_data, valid_trgt, config.num_epochs, config.batch_size, [reduce_lr,  test_callback, tensorboard])
        
        ## Save Results
        elapsed_times_all[b_id] = timer() - one_timer
        my_network.save_statistic_results(saving_dir, elapsed_times_all)
        my_network.save_img_results(saving_dir, test_data, original_test_trgt, config.num_epochs)
        
        my_netowrk = None
