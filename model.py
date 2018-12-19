import os, time, h5py, argparse, sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import sqrt

import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from keras import layers
from keras.initializers import RandomNormal, lecun_uniform
from keras.losses import categorical_crossentropy

from tensorflow.python.ops import array_ops
from tensorflow.python.summary import summary as tf_summary

from UNet import UNet, Saliency_UNet, Dilated_Saliency_UNet

class Net_Model:
    
    def __init__(self, layer_level, input_shape, Loss, lr, num_chn=3, num_class=2, VISUALISATION=False, SALIENCY=False, DILATION=False):
        self.layer_level = layer_level
        self.input_shape = input_shape
        self.num_class = num_class
        self.train_setting(Loss)
        
        if DILATION and SALIENCY:
            self.model = Dilated_Saliency_UNet(input_shape, num_chn, lr, self.loss, self.activation)
            self.net_name ='Dilated_Saliency_UNet'
        elif SALIENCY:
            self.model = Saliency_UNet(input_shape, num_chn, lr, self.loss, self.activation)
            self.net_name ='Saliency_UNet'
        else:
            input_shape = list(input_shape)
            input_shape[2] = num_chn
            input_shape = tuple(input_shape)
            self.model = UNet(input_shape, self.layer_level, lr, self.loss, self.activation)
            self.net_name ='UNet_depth'+str(self.layer_level)
        
        self.model = self.model.compiled_network()
        
        if VISUALISATION: 
            self.visualise()
                
    def train_setting(self, Loss):
        if Loss == 'crossentropy':
            self.loss = categorical_crossentropy
            self.activation = 'softmax'
            self.num_class = 3
        elif Loss == 'dice_coef':
            self.loss = dice_coef_loss
            self.activation = 'sigmoid'
        elif Loss == 'gdl':
            self.loss = GDL
            self.activation = 'sigmoid'
        elif Loss == 'iou':
            self.loss = iou_loss
            self.activation = 'softmax'
            self.num_class = 3
        elif Loss == 'iou_crossentropy':
            self.loss = iou_crossentropy_loss
            self.activation = 'softmax'
            self.num_class = 3
        else:
            print('Loss should be either \'crossentropy\' or \'dice_coef\'')
            sys.exit(1)
        
    def restore(self, weights_path):
        self.model.load_weights(weights_path)
        
        
    def visualise(self):
        def put_kernels_on_grid (kernel, shape, pad = 1):
            print(shape)
            shape = np.array([shape[1],shape[2],shape[0],shape[3]])
            kernel = tf.transpose(kernel,(1,2,0,3))
            def factorization(n):
                for i in range(int(sqrt(float(n))), 0, -1):
                    if n % i == 0:
                        if i == 1: print('Who would enter a prime number of filters')
                        return (i, int(n / i))

            (grid_Y, grid_X) = factorization (shape[3])
            print ('grid: %d = (%d, %d)' % (shape[3], grid_Y, grid_X))

            # pad X and Y
            x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

            # X and Y dimensions, w.r.t. padding
            Y = shape[0] + 2 * pad
            X = shape[1] + 2 * pad

            channels = shape[2]

            # put NumKernels to the 1st dimension
            x = tf.transpose(x, (3, 0, 1, 2))
            # organize grid on Y axis
            x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

            # switch X and Y axes
            x = tf.transpose(x, (0, 2, 1, 3))
            # organize grid on X axis
            x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

            # back to normal order (not combining with the next step for clarity)
            x = tf.transpose(x, (2, 1, 3, 0))
            # to tf.image_summary order [batch_size, height, width, channels],
            #   where in this case batch_size == 1
            x = tf.transpose(x, (3, 0, 1, 2))

            # scaling to [0, 255] is not necessary for tensorboard
            return x
        
        prev_layer = None
        for idx, layer in enumerate(self.model.layers):
            
            if 'input' in layer.name:
                input_i = layer.input
                input_img = array_ops.squeeze(input_i)
                img = input_img[0,:,:]
                #img = array_ops.transpose(img, perm=[2,0,1])
                #img = tf.concat([img[0],img[1]],axis=1)
                img = tf.expand_dims(img, 0)
                img = tf.expand_dims(img, 3)
                tf_summary.image(layer.name, img)
            elif 'maxpool' in layer.name or 'up2d' in layer.name:
                shape = np.concatenate([np.array([1]),self.shape_list.pop(0)])
                print(layer.name,shape)
                output = layer.output
                output_name = layer.name.replace(':','_')
                output_img = array_ops.squeeze(output)
                # pick the output of one slice
                output_img = output_img[0,:,:,:]
                output_img = tf.expand_dims(output_img, 0)
                output_img = put_kernels_on_grid(output_img, shape)
                tf_summary.image(output_name, output_img)
            elif idx == len(self.model.layers)-1:
                output = layer.output
                output_name = layer.name.replace(':','_')
                output_img = array_ops.squeeze(output)
                output_img = output_img[0,:,:,:]
                output_img = array_ops.transpose(output_img, perm=[2,0,1])
                output_img = tf.concat([output_img[0],output_img[1],output_img[2]], axis=1)
                img = tf.expand_dims(output_img, 0)
                img = tf.expand_dims(img, 3)
                tf_summary.image(output_name, img)
            elif 'conv' in layer.name:
                chn = layer.filters
           
    
    def initialize(self):
        keras.backend.get_session().run(tf.global_variables_initializer())
    
    def train(self, train_data, train_trgt, val_split, epochs, nb_samples, callbacks):

        self.history_callback = self.model.fit(x = train_data, 
                                        y = train_trgt,
                                        validation_split=val_split,
                                        epochs=epochs,
                                        batch_size=nb_samples,
                                        shuffle=True,
                                        callbacks=callbacks)
    
    def predict(self, test_data, test_trgt):
        print('Test data number : ',str(test_data.shape[0]),' Slices')        
        test_data_img = []
        test_trgt_img = []
        for i in range(10):
            test_data_img.append(test_data[20+35*i])
            test_trgt_img.append(test_trgt[20+35*i])
            print('Slice ',i,' Number of WMH voxels: GT - ',np.sum(np.where(test_trgt[20+35*i]>0)))
        
        test_data_img = np.array(test_data_img)
        test_trgt_img = np.array(test_trgt_img)
        test_pred_img = self.model.predict(test_data_img, verbose=1)
        
        return test_data_img, test_trgt_img, test_pred_img
        
    def save_statistic_results(self, saving_dir, elapsed_times_all):
        ## SAVE ELAPSED TIME
        #f = open(saving_dir+'history_elapsedTime.txt', 'ab')
        #np.savetxt(f,elapsed_times_all)
        #f.close()

        ## SAVING TRAINED MODEL AND WEIGHTS
        self.model.save_weights(saving_dir+'train_models.h5')
        model_json = self.model.to_json()
        with open(saving_dir+'train_models.json', "w") as json_file:
            json_file.write(model_json)

        ## SAVE HISTORY
        for h in self.history_callback.history:
            numpy_history = np.array(self.history_callback.history[h])
            f = open(saving_dir+h+'.txt','ab')
            np.savetxt(f,numpy_history)
            f.close()
    
    def save_img_results(self, saving_dir, test_data, test_trgt):
        print('Test data number : ',str(test_data[0].shape[0]),' Slices') 
        '''
        test_data = np.concatenate(test_data, axis=3)
        test_data_img = []
        test_trgt_img = []
        for i in range(10):
            test_data_img.append(test_data[20+35*i])
            test_trgt_img.append(test_trgt[20+35*i])
            print('Slice ',i,' Number of WMH voxels: GT - ',np.sum(np.where(test_trgt[20+35*i]==2)))
        
        test_data_img = np.array(test_data_img)
        num_chn = test_data_img.shape[-1]
        data_img = [np.expand_dims(test_data_img[:,:,:,idx], axis=3) for idx in range(num_chn)]
        test_trgt_img = np.array(test_trgt_img)
        '''
        
        test_pred_img = self.model.evaluate(test_data, test_trgt, verbose=0)#model.predict(test_data, verbose=1)
        
        chn = ['FLAIR','IAM', 'T1W']
        
        fig_idx = 0
        fig, ax = plt.subplots(10, num_chn+2)
        fig.set_size_inches((num_chn+2)*5, 50)

        for data_img, pred_img, gt_img in zip(test_data_img, test_pred_img, test_trgt_img):
            print(data_img.shape, pred_img.shape, gt_img.shape)
            gt_img = (gt_img[:,:,0]).astype(np.float)
            # Convert class probability map to labels
            if pred_img.shape[-1]>1:
                pred_img = (np.argmax(pred_img, axis=2)).astype(np.float)
            else:
                pred_img = (pred_img[:,:,0].astype(np.float))

            for j in range(num_chn):
                flair_img = (data_img[:,:,j]).astype(np.float)
                cmap='gray'
                if j==1 : cmap='jet'
                ax0 = ax[fig_idx, j].imshow(flair_img, cmap=cmap)
                ax[fig_idx, j].set_title(chn[j])
                divider0 = make_axes_locatable(ax[fig_idx, j])
                cax0 = divider0.append_axes("right", size="7%", pad=0.05)
                fig.colorbar(ax0, cax=cax0)

            ax1 = ax[fig_idx, num_chn].imshow(gt_img, cmap='jet', vmin=0, vmax=self.num_class-1 )
            ax[fig_idx, num_chn].set_title('Ground Truth')
            divider1 = make_axes_locatable(ax[fig_idx, num_chn])
            cax1 = divider1.append_axes("right", size="7%", pad=0.05)
            fig.colorbar(ax1, cax=cax1)

            ax2 = ax[fig_idx, num_chn+1].imshow(pred_img, cmap='jet', vmin=0, vmax=self.num_class-1)
            ax[fig_idx, num_chn+1].set_title('Prediction')
            divider2 = make_axes_locatable(ax[fig_idx, num_chn+1])
            cax2 = divider2.append_axes("right", size="7%", pad=0.05)
            fig.colorbar(ax2, cax=cax2)

            fig_idx = fig_idx+1

        fig.savefig(saving_dir+'test_result_img.png')
        
    def layer_init(self, init_num):
        #Initialise layers
        model_len = len(self.model.layers)-1
        print('==================== Initialised Layers ======================')
        session = keras.backend.get_session()
        for i in range(init_num):
            idx = i
            layer = self.model.layers[idx]
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = lecun_uniform
                layer.kernel.initializer.run(session=session)
                print(layer.name)
    
        
    def summary(self):
        self.model.summary()
        self.model.get_config()
        
