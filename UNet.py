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

class UNet:
    def __init__(self, input_shape, depth, lr, loss, activation):
        self.network = self.building_net(input_shape, depth, num_class=3, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
        
        return self.network
            
    def building_net(self, input_shape, depth, num_class, activation):
        conv = inputs = Input(input_shape, name='input_layer')
        conv_stack = []
        
        ## Encoding Part
        for d in range(depth):
            if d == 0: filter_size = (5,5)
            else: filter_size=(3,3)
                
            conv = self.conv_block('relu', 64*(d+1), filter_size, conv, depth_num=str(d)+'_0')
            conv = self.conv_block('relu', 64*(d+1), filter_size, conv, depth_num=str(d)+'_1')
            conv_stack.append(conv)
            conv = MaxPooling2D(pool_size=(2,2), name='maxpool2d_'str(d))
        
        ## Decoding Part
        for d in range(depth, 2*depth+1):
            conv = self.conv_block('relu', 64*(d+1), (3,3), conv, depth_num=str(d)+'_0')
            conv = Dropout(0.25, name='do_'+str(d))(conv)
            conv = self.conv_block('relu', 64*(d+1), (3,3), conv, depth_num=str(d)+'_1')
            if d == 2*depth: break
            conv = UpSampling2D(pool_size=(2,2), name='up2d_'+str(d))(conv)
            conv = concatenate([conv, conv_stack.pop()], axis=-1, name='concat_'+str(d))
        
        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), padding='same', activation=activation, name='conv2d_'+str(depth*2))(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def conv_block(self, activation, filters, filter_size, input_layer, depth_num, modal_name="", dilation=1):
        conv = Conv2D(filters, filter_size, padding='same',
                      name='conv2d_'+modal_name+depth_num,
                      dilation_rate = dilation)(input_layer)
        conv = BatchNormalization(name='bn_'+modal_name+depth_num)(conv)
        if activation=='relu':
            conv = ReLU(name='relu_'+modal_name+depth_num)(conv)
        else:
            conv = LeakyReLU(alpha=0.5, name='Lrelu_'+modal_name+depth_num)(conv)
            

class Saliency_UNet(UNet):
    
    def __init__(self, input_shape, num_modal, lr, loss, activation):
        
        self.network = self.building_net(input_shape, num_modal, num_class=3, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
        
        return self.network
    
    def building_net(self, input_shape, num_modal, activation, DILATION=False):
        modal_order = ['flair', 'iam', 't1w']
        inputs = []
        conv_stack = []
        skip_layers1 = []
        skip_layers2 = []
        concat_layers = []
        
        for m in range(num_modal):
            modal = modal_order[m]
            input_layer = Input(input_shape, name=modal+'_layer')
            inputs.append(input_layer)
            
            if m==2: activation = 'Lrelu'
            else: activation = 'relu'
            if DILATION: d1,d2 =[4,2],[2,1]
            else: d1, d2=[1,1]
                
            conv_1, pool_1 = self.encoder_block(inputs, activation, 
                                                modal[0], 64, (5,5), 0, d1)
            conv_2, pool_2 = self.encoder_block(pool_1, activation, 
                                                modal[0], 128, (3,3), 1, d2)
            skip_layers1.append(conv_1)
            skip_layers2.append(conv_2)
            concat_layers.append(pool_2)

        conv = self.decoder_block(concat_layers, 256, (3,3), 2)
        conv = self.decoder_block([conv]+skip_layers2, 128, (3,3), 3)
        conv = self.decoder_block([conv]+skip_layers1, 64, (3,3), 4)
        
        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), padding='same', activation=activation, name='conv2d_5')(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def encoder_block(self, inputs, activation, modality, filters, filter_size, depth, dilation=[1,1]):
        
        depth = modality+'_'str(depth)
        conv = super().conv_block(activation, filters, filter_size, inputs, depth_num = depth+'_0_d'+str(dilation[0]), dilation=dilation[0])
        conv = super().conv_block(activation, filters, filter_size, conv, depth_num = depth+'_1_d'+str(dilation[1]), dilation=dilation[1])
        
        if modality == 't':
            pool = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_'+depth)(conv)
        else:
             pool = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+depth)(conv)
                
        return conv, pool
    
    def decoder_block(self, concat, filters, filter_size, depth):
        d = str(depth)
        conv = concatenate(concat, axis=-1, name='concat_'+d)
        conv = super().conv_block('relu', filters, filter_size, conv, depth_num=d+'_0')
        conv = Dropout(0.25, name='do_'+d)(conv)
        conv = super().conv_block('relu', filters, filter_size, conv, depth_num=d+'_1')
        conv = UpSampling2D(pool_size=(2,2), name='up2d_'+d)(conv)
        
        return conv

class Dilated_Saliency_UNet(Saliency_UNet):
    
    def __init__(self, input_shape, num_modal, lr, loss, activation):
        self.network = super().building_net(input_shape, num_modal, num_class=3, activation=activation, DIALTION=True)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
        
        return self.network
    
    