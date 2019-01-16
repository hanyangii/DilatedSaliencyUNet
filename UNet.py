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

from UNet_lib import dice_coef

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
                
            conv = self.conv_block('relu', 64*(2**d), filter_size, conv, padding="CONSTANT", depth_num=str(d)+'_0')
            conv = self.conv_block('relu', 64*(2**d), filter_size, conv, padding="CONSTANT", depth_num=str(d)+'_1')
            conv_stack.append(conv)
            depth_stack.append(64*(2**d))
            conv = MaxPooling2D(pool_size=(2,2), name='maxpool2d_'+str(d))(conv)
        
        ## Decoding Part
        for d in range(depth, 2*depth+1):
            if d == depth: cur_depth = 64*(2**d)
            else: cur_depth = depth_stack.pop()
            conv = self.conv_block('relu', cur_depth , (3,3), conv, padding="CONSTANT", depth_num=str(d)+'_0')
            conv = Dropout(0.25, name='do_'+str(d))(conv)
            conv = self.conv_block('relu', cur_depth , (3,3), conv, padding="CONSTANT", depth_num=str(d)+'_1')
            if d == 2*depth: break
            conv = UpSampling2D(size=(2,2), name='up2d_'+str(d))(conv)
            conv = concatenate([conv, conv_stack.pop()], axis=-1, name='concat_'+str(d))
        
        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), activation=activation, padding="same", name='conv2d_'+str(depth*2))(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def conv_block(self, activation, filters, filter_size, input_layer, depth_num, padding, modal_name="", dilation=1):
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
    
    def __init__(self, input_shape, num_modal, num_class, lr, loss, activation):
        
        self.network = self.building_net(input_shape, num_modal,num_class, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
            
    def building_net(self, input_shape, num_modal, num_class, activation, DILATION=False):
        modal_order = ['flair', 'iam', 't1w']
        inputs = []
        conv_stack = []
        skip_layers1 = []
        skip_layers2 = []
        concat_layers = []
        
        for m in range(num_modal):
            modal = modal_order[m]
            print(modal)
            input_layer = Input(input_shape, name=modal+'_layer')
            inputs.append(input_layer)
            
            if m==2: modal_activation = 'Lrelu'
            else: modal_activation = 'relu'
            if DILATION : 
                #Basic dilation factors 1-2-4-2
                print('Dilation on ',modal)
                d1 = [4,2]
                d2 = [2,1]
            else: 
                d1 = d2=[1,1]
                
            conv_1, pool_1 = self.encoder_block(input_layer, modal_activation, 
                                                modal[0], 64, (5,5), 0, d1)
            conv_2, pool_2 = self.encoder_block(pool_1, modal_activation, 
                                                modal[0], 128, (3,3), 1, d2)
            skip_layers1.append(conv_1)
            skip_layers2.append(conv_2)
            concat_layers.append(pool_2)

        conv = self.decoder_block(concat_layers, 256, (3,3), 2, True)
        conv = self.decoder_block([conv]+skip_layers2, 128, (3,3), 3, True)
        conv = self.decoder_block([conv]+skip_layers1, 64, (3,3), 4, False)
        
        ## Fully Connected
        fconv = Conv2D(num_class, (1,1), activation=activation, padding="same", name='conv2d_5')(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def encoder_block(self, inputs, activation, modality, filters, filter_size, depth, dilation=[1,1]):
        
        depth = modality+'_'+str(depth)
        conv = super().conv_block(activation, filters, filter_size, inputs, padding="CONSTANT", depth_num = depth+'_0_d'+str(dilation[0]), dilation=dilation[0])
        conv = super().conv_block(activation, filters, filter_size, conv, padding="CONSTANT", depth_num = depth+'_1_d'+str(dilation[1]), dilation=dilation[1])
        
        if modality == 't':
            pool = AveragePooling2D(pool_size=(2, 2), name='avgpool2d_'+depth)(conv)
        else:
             pool = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_'+depth)(conv)
                
        return conv, pool
    
    def decoder_block(self, concat, filters, filter_size, depth, UP_SAMPLE):
        d = str(depth)
        conv = concatenate(concat, axis=-1, name='concat_'+d)
        conv = super().conv_block('relu', filters, filter_size, conv, padding="CONSTANT", depth_num=d+'_0')
        conv = Dropout(0.25, name='do_'+d)(conv)
        conv = super().conv_block('relu', filters, filter_size, conv, padding="CONSTANT", depth_num=d+'_1')
        if UP_SAMPLE:
            conv = UpSampling2D(size=(2,2), name='up2d_'+d)(conv)
        
        return conv

class Noppoling_Dilated_Saliency_UNet(UNet):
    
    def __init__(self, input_shape, num_modal, num_class, lr, loss, activation):
        
        self.network = self.building_net(input_shape, num_modal,num_class, activation=activation)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
            
    def building_net(self, input_shape, num_modal, num_class, activation, DILATION=False):
        modal_order = ['flair', 'iam', 't1w']
        inputs = []
        conv_stack = []
        skip_layers1 = []
        concat_layers = []
        
        for m in range(num_modal):
            modal = modal_order[m]
            print(modal)
            input_layer = Input(input_shape, name=modal+'_input_layer')
            inputs.append(input_layer)
            
            if m==2: modal_activation = 'Lrelu'
            else: modal_activation = 'relu'
                
            conv_1 = self.encoder_block(input_layer, modal_activation, 
                                                modal[0], [64,128], (5,5), 0, [1,2])
            conv_2 = self.encoder_block(conv_1, modal_activation, 
                                                modal[0], [256,256], (3,3), 1, [4,2])
           
            skip_layers1.append(conv_1)
            concat_layers.append(conv_2)

        conv = self.decoder_block(concat_layers, [256, 256], [(3,3),(5,5)], 2, [2,1])
        conv = self.decoder_block([conv]+skip_layers1, [128,64] , [(3,3),(3,3)], 3, [1,1])
        #conv = self.decoder_block(conv, 64, [(3,3),(3,3)], 4, [1,1])
        
        ## Fully Connected
        #conv = super().conv_block('relu', 256, (3,3), conv, depth_num='5_0_d1', dilation=1)
        fconv = Conv2D(num_class, (1,1), activation=activation, padding="same", name='conv2d_5')(conv)
        
        model = Model(input=inputs, output=fconv)
        
        return model
    
    def encoder_block(self, inputs, activation, modality, filters, filter_size, depth, dilation=[1,1]):
        
        depth = modality+'_'+str(depth)
        conv = super().conv_block(activation, filters[0], filter_size, inputs, padding="CONSTANT", depth_num = depth+'_0_d'+str(dilation[0]), dilation=dilation[0])
        conv = super().conv_block(activation, filters[1], filter_size, conv, padding="CONSTANT", depth_num = depth+'_1_d'+str(dilation[1]), dilation=dilation[1])
        

        return conv 
    
    def decoder_block(self, concat, filters, filter_size, depth, dilation):
        d = str(depth)
        if isinstance(concat, list):
            conv = concatenate(concat, axis=-1, name='concat_'+d)
        else:
            conv = concat
        conv = super().conv_block('relu', filters[0], filter_size[0], conv, padding="CONSTANT", depth_num=d+'_0_d'+str(dilation[0]), dilation=dilation[0])
        conv = Dropout(0.25, name='do_'+d)(conv)
        conv = super().conv_block('relu', filters[1], filter_size[1], conv, padding="CONSTANT", depth_num=d+'_1_d'+str(dilation[1]), dilation=dilation[1])
        
        return conv    
    
class Dilated_Saliency_UNet(Saliency_UNet):
    
    def __init__(self, input_shape, num_modal, num_class, lr, loss, activation):
        self.network = super().building_net(input_shape, num_modal,num_class, activation=activation, DILATION=True)
        self.network.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[categorical_accuracy, dice_coef])
     

    

    
    
