import os, time, h5py, argparse, sys, shutil

import scipy.io as sio
import numpy as np
import tensorflow as tf
import nibabel as nib

import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K

from timeit import default_timer as timer

from utils import TestCallback, LossHistory
from model import Net_Model

def test_model(config, test_data, test_trgt,net_depth, SALIENCY, DILATION, restore_dir, net_type, label_list, affine_list, dilation_factor=None):
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
                            DILATION=DILATION,
                            dilation_factor=dilation_factor)
    
    '''Restore Weights'''
    my_network.restore(restore_dir)
    save_dir = restore_dir.split('train_models.h5')[0]
    if os.path.exists(save_dir+'predictions1'): shutil.rmtree(save_dir+'predictions1')
    os.mkdir(save_dir+'predictions1')
    
    test_trgt = to_categorical(test_trgt, num_classes = my_network.num_class)
    
    '''Test each volume'''
    with open(save_dir+'/predictions1/test_slice_dsc.txt', 'w') as dsc_file:
        with open(save_dir+'/predictions1/brain_dsc.txt', 'w') as brain_dsc_file:
            for i, subject in enumerate(label_list):
                prediction = []
                subject = subject.split('IAM/')[-1]
                subject = subject.split('/WMH')[0]
                affine = affine_list[i]
                print('Predict : ', subject)
                for j in range(5):
                    trgt = test_trgt[i*35+7*j:i*35+7*(j+1),:,:,:]
                    if isinstance(test_data, list):
                        test_img = [modal_data[i*35+7*j:i*35+7*(j+1),:,:,:] for modal_data in test_data]
                    else: 
                        test_img = test_data[i*35+7*j:i*35+7*(j+1),:,:,:]

                    # Predict
                    slice_dsc, pred_img = my_network.test_result_store(test_img, trgt)
                    prediction.append(pred_img)
                    for dsc in slice_dsc:
                        dsc = str(dsc)[1:-1]+'\n'
                        dsc_file.write(str(dsc))

                prediction = np.concatenate(prediction, axis = 0)
                
                
                #Whole Brain DSC Calculation
                gt = np.where(np.argmax(test_trgt[i*35:(i+1)*35,:,:,:], axis=-1)==2, 1, 0)
                print('WMH prediction and label data shape : ', np.shape(prediction), np.shape(gt))
                smooth=1e-7
                wmh_vol = np.sum(gt)
                brain_dsc = (2. * np.sum(gt*prediction)+smooth) / (np.sum(gt) + np.sum(prediction)+smooth)
                
                #Sensitivity
                sensitivity = np.sum(gt * prediction)/np.sum(gt)
                gt_n = (~gt.astype(bool)).astype(float)
                fp_rate = np.sum(gt_n * prediction)/np.sum(gt_n)
                
                line = str((wmh_vol, sensitivity, fp_rate, brain_dsc))[1:-1]+'\n'
                brain_dsc_file.write(line)
                
                prediction = np.moveaxis(prediction, 0, -1)
                prediction = nib.Nifti1Image(prediction, affine, nib.Nifti1Header())
                print('nii data shape: ', prediction.header.get_data_shape())
                nib.save(prediction, save_dir+'predictions1/'+subject+'.nii.gz')
                
                
    print('Total Test Time: ', my_network.test_time)
        

def train_model(config, START_TIME, net_depth, SALIENCY, DILATION, restore_dir, net_type, train_dat, test_dat, dilation_factor=None):
    
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
                                DILATION=DILATION,
                                dilation_factor=dilation_factor)
        
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
        
        # to_categorical for labels
        if my_network.activation == 'softmax':
            train_trgt = to_categorical(train_trgt, num_classes = my_network.num_class)
            original_test_trgt = test_trgt
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
        my_network.train(train_data, train_trgt, config.num_epochs, config.batch_size, [reduce_lr,  test_callback, tensorboard])
        
        ## Save Results
        elapsed_times_all[b_id] = timer() - one_timer
        my_network.save_statistic_results(saving_dir, elapsed_times_all)
        my_network.save_img_results(saving_dir, test_data, original_test_trgt, config.num_epochs)
        
        my_netowrk = None
