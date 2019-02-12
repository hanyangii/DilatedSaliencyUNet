import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization
from keras.layers import AveragePooling2D, UpSampling2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.layers import LeakyReLU, ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate

from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import categorical_accuracy, binary_accuracy

from utils import dice_coef

class UNet:
    def __init__(self, input_shape, depth, lr, loss, activation):
        self.network = self.building_net(input_shape, depth, num_class=3, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
    
    def compiled_network(self):
        return self.network
        
            
    def building_net(self, input_shape, depth, num_class, activation):
        conv = inputs = Input(input_shape, name='input_layer')
        conv_stack = []
        depth_stack = []
        
        ## Encoding Part
        for d in range(depth):
            if d == 0: filter_size = (5,5)
            else: filter_size=(3,3)
                
            conv = self.conv_block('relu', 64*(2**d), filter_size, conv, depth_num=str(d)+'_0')
            conv = self.conv_block('relu', 64*(2**d), filter_size, conv, depth_num=str(d)+'_1')
            conv_stack.append(conv)
            depth_stack.append(64*(2**d))
            conv = MaxPooling2D(pool_size=(2,2), name='maxpool2d_'+str(d))(conv)
        
        ## Decoding Part
        for d in range(depth, 2*depth+1):
            if d == depth: cur_depth = 64*(2**d)
            else: cur_depth = depth_stack.pop()
            conv = self.conv_block('relu', cur_depth , (3,3), conv, depth_num=str(d)+'_0')
            conv = Dropout(0.25, name='do_'+str(d))(conv)
            conv = self.conv_block('relu', cur_depth , (3,3), conv, depth_num=str(d)+'_1')
            if d == 2*depth: break
            conv = UpSampling2D(size=(2,2), name='up2d_'+str(d))(conv)
            conv = concatenate([conv, conv_stack.pop()], axis=-1, name='concat_'+str(d))
        
        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), activation=activation, padding="same", name='conv2d_'+str(depth*2))(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def conv_block(self, activation, filters, filter_size, input_layer, depth_num, modal_name="", dilation=1):
        conv = Conv2D(filters, filter_size,
                      name='conv2d_'+modal_name+depth_num, padding="same",
                      dilation_rate = dilation)(input_layer)
        conv = BatchNormalization(name='bn_'+modal_name+depth_num)(conv)
        if activation=='relu':
            conv = ReLU(name='relu_'+modal_name+depth_num)(conv)
        else:
            conv = LeakyReLU(alpha=0.5, name='Lrelu_'+modal_name+depth_num)(conv)
        
        return conv
            

class Saliency_UNet(UNet):
    
    def __init__(self, input_shape, depth, num_modal, num_class, lr, loss, activation):
        
        self.network = self.building_net(input_shape, depth, num_modal,num_class, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
            
    def building_net(self, input_shape, depth, num_modal, num_class, activation, DILATION=False, dilation_factor=None):
        modal_order = ['flair', 'iam', 't1w']
        inputs = []
        filter_stack = []
        skip_layers = [[] for i in range(depth)]
        concat_layers = []
        
        for m in range(num_modal):
            modal = modal_order[m]
            print(modal)
            pool = Input(input_shape, name=modal+'_layer')
            inputs.append(pool)
            
            if m==2: modal_activation = 'Lrelu'
            else: modal_activation = 'relu'
            if not DILATION : 
                dilation_factor=[[1,1] for i in range(depth)]
            
            for d in range(depth):
                if d==0 and DILATION == False: filter_size=(5,5)
                else: filter_size = (3,3)
                conv, pool = self.encoder_block(pool, modal_activation, modal[0], 64*(2**d), 
                                                filter_size, d, dilation_factor[d])
                skip_layers[d].append(conv)
                filter_stack.append(64*(2**d))
            concat_layers.append(pool)
        
        for d in range(depth, 2*depth+1):
            if d == depth:
                filter_num = 64*(2**d)
                concat = concat_layers
            else:
                filter_num = filter_stack.pop()
                concat = [conv] + skip_layers.pop()
            
            do = True
            if d == 2*depth: do = False
            
            conv = self.decoder_block(concat, filter_num, (3,3), d, do)

        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), activation=activation, padding="same", name='conv2d_'+str(depth*2+1))(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def encoder_block(self, inputs, activation, modality, filters, filter_size, depth, dilation=[1,1]):
        
        depth = modality+'_'+str(depth)
        conv = super().conv_block(activation, filters, filter_size, inputs, depth_num = depth+'_0_d'+str(dilation[0]), dilation=dilation[0])
        conv = super().conv_block(activation, filters, filter_size, conv, depth_num = depth+'_1_d'+str(dilation[1]), dilation=dilation[1])
        
        if modality == 't':
            pool = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_'+depth)(conv)
        else:
             pool = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+depth)(conv)
                
        return conv, pool
    
    def decoder_block(self, concat, filters, filter_size, depth, UP_SAMPLE):
        d = str(depth)
        conv = concatenate(concat, axis=-1, name='concat_'+d)
        conv = super().conv_block('relu', filters, filter_size, conv, depth_num=d+'_0')
        conv = Dropout(0.25, name='do_'+d)(conv)
        conv = super().conv_block('relu', filters, filter_size, conv, depth_num=d+'_1')
        if UP_SAMPLE:
            conv = UpSampling2D(size=(2,2), name='up2d_'+d)(conv)
        
        return conv
  
    
class Dilated_Saliency_UNet(Saliency_UNet):
    
    def __init__(self, input_shape, depth, num_modal, num_class, lr, loss, activation, dilation_factor):
        self.network = super().building_net(input_shape, depth, num_modal,num_class, activation=activation, DILATION=True, dilation_factor=dilation_factor)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
     

    

    
    
