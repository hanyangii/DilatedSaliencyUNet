import os, time, h5py, argparse, sys

import scipy.io as sio
import numpy as np
import tensorflow as tf

import keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from skimage.util.shape import view_as_windows
from timeit import default_timer as timer

from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import Model, model_from_json
from keras.initializers import RandomNormal, lecun_uniform

from data import *

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

from UNet_lib import *
print_options(vars(args))

if __name__ == '__main__':
    
    # Parameter Setting
    shuffle_epoch = args.shuffle_epoch
    lr            = args.lr
    nb_samples    = args.nb_samples
    num_epochs    = args.num_epochs
    bn_momentum   = args.bn_momentum
    TRSH          = args.TRSH
    num_chn       = args.num_chn
    fold          = args.fold
    random_num    = args.random_num
    init_num      = args.init_num
    img_x, img_y, img_z = args.img_size[0], args.img_size[1], args.img_size[2]
    FINETUNING    = args.finetune
    PRETRAINING   = args.pretrain
    TRAINING      = False
    if not(FINETUNING or PRETRAINING): 
        TRAINING = True
    elif (FINETUNING and PRETRAINING):
        print('You cannot choose both finetuning and pretraining modes')
        sys.exit(1)
    
    if (FINETUNING or TRAINING):
        n_class   = 3
    else:
        n_class   = 6
    
    #Loss 
    if args.loss == 'crossentropy':
        Loss = categorical_crossentropy
    elif args.loss == 'dice_coef':
        Loss = dice_coef_loss
    elif args.loss == 'gdl':
        Loss = GDL
    else:
        print('Loss should be either \'crossentropy\' or \'dice_coef\'')
        sys.exit(1)
        
    
    #Data path
    if FINETUNING:
        args.dir_name = 'Finetune_'+args.dir_name
        pretrained_weights_path = './results/Compare_ShallowFinetuning/Pretrain___UNet_depth3_BN_beforeRELU_DO025_20181113-0829_64x64x1_chn1_cv0_ep50/train_models.h5'
        pretrained_model_path = './results/Compare_ShallowFinetuning/Pretrain___UNet_depth3_BN_beforeRELU_DO025_20181113-0829_64x64x1_chn1_cv0_ep50/train_models.json'
        train_config_dir = 'data/com_test_configs_2fold_adni60'
        test_config_dir = 'data/com_test_configs_2fold_adni60'
    elif PRETRAINING:
        args.dir_name = 'Pretrain_'+args.dir_name
        train_config_dir = 'data/com_train_configs_2fold_adni60'
        test_config_dir = 'data/com_train_configs_2fold_adni60'
    elif TRAINING:
        args.dir_name = 'Training_'+args.dir_name
        train_config_dir = 'data/com_test_configs_2fold_adni60'
        test_config_dir = 'data/com_test_configs_2fold_adni60'


    elapsed_times_all = np.zeros((fold))
    for b_id in range(fold):  
        
        '''Building Network'''
        net_name = define_net_name(args, START_TIME)
       
        '''Fine-tuning'''
        if FINETUNING:
            #Load pre-trained model
            json_model_file = open(pretrained_model_path, 'r')
            loaded_model = json_model_file.read()
            json_model_file.close()
            pretrained_network = model_from_json(loaded_model)
            pretrained_network.load_weights(pretrained_weights_path)
            pretrained_network.layers.pop()
            
            #Convert last layer
            new_out_layer = Conv2D(n_class, (1, 1), padding='same', activation='softmax', name='conv2d_9_18')(pretrained_network.layers[-1].output)
            my_network = Model(input = pretrained_network.input, output=new_out_layer)
            
            #Build a new network 
            my_network.compile(optimizer=Adam(lr=lr), loss=Loss, metrics=[categorical_accuracy, dice_coef])
            
            
            #Initialise layers
            model_len = len(my_network.layers)-1
            print('======================== Initialised Layers ========================')
            session = keras.backend.get_session()
            for i in range(init_num):
                idx = i
                layer = my_network.layers[idx]
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel_initializer = lecun_uniform
                    layer.kernel.initializer.run(session=session)
                    print(layer.name)

        else:
            my_network = Building_UNet(args.depth, args.batch_norm, (None, None, num_chn), Loss=Loss, lr=lr, num_class=n_class, PRETRAIN=PRETRAINING)
            keras.backend.get_session().run(tf.global_variables_initializer())
        
        print("\nBuilding network #1..")
        history_batch = LossHistory()
        my_network.summary()
        my_network.get_config()
        
        patch_size = str(img_x) + 'x' + str(img_y) + 'x' + str(img_z) + '_chn' + str(num_chn)
        saving_filename = str(net_name) +'_'+str(patch_size) + '_cv' + str(b_id) + '_ep' + str(num_epochs)
        print('\n\nSaving_filename: ' + saving_filename)
        win_shape = (img_x, img_y, img_z)
        
        '''Load Data'''
        if PRETRAINING:
            train_data, train_trgt = generate_train_data(train_config_dir, 0, TRSH, win_shape, random_num, num_chn)
            test_data, test_trgt = generate_test_data(test_config_dir, 1, random_num, num_chn)
        else:
            train_data, train_trgt = generate_finetune_train_data(train_config_dir, b_id, TRSH, win_shape, random_num, num_chn)
            test_data, test_trgt = generate_finetune_test_data(test_config_dir, 1, random_num, num_chn)
        
        print(train_trgt.shape, ' class num: ', n_class,' max val:',np.max(train_trgt))
        
        train_trgt_cate = to_categorical(train_trgt,n_class)
        test_trgt_cate = to_categorical(test_trgt,n_class)
        
        print('\nTRAINING DATASET PERMUTED size: ' + str(train_data.shape))
        print('TRAINING LABEL DATASET PERMUTED size: ' + str(train_trgt_cate.shape))
        print('\nVALIDATION DATASET PERMUTED size: ' + str(test_data.shape))
        print('VALIDATION LABEL DATASET PERMUTED size: ' + str(test_trgt_cate.shape))
        print("\nSaving filename: " + saving_filename)
        
        one_timer = timer()
        
        # Tensorboard Visualisation
        saving_dir = './results/'+saving_filename+'/'
        
        if os.path.exists(saving_dir):
            os.rmdir(saving_dir)
        os.mkdir(saving_dir)
        
        # History Callbacks
        tensorboard = TensorBoard(log_dir = saving_dir + 'tensorboard_log/', batch_size = nb_samples) 
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience, min_lr=2e-10)
        
        # Train network
        history_callback = my_network.fit(train_data, train_trgt_cate,
            epochs=num_epochs,
            batch_size=nb_samples,
            shuffle=True,
            validation_data=(test_data, test_trgt_cate),
            callbacks=[reduce_lr,tensorboard])

        ## SAVE ELAPSED TIME
        elapsed_times_all[b_id] = timer() - one_timer
        save_results(saving_dir, elapsed_times_all, my_network, history_callback)
        
        ## SAVE SAMPLE RESULTS BY IMAGE
        print('Test data number : ',str(test_data.shape[0]),' Slices')        
        test_data_img = []
        test_trgt_img = []
        for i in range(10):
            test_data_img.append(test_data[20+35*i])
            test_trgt_img.append(test_trgt[20+35*i])
            print('Slice ',i,' Number of WMH voxels: GT - ',np.sum(np.where(test_trgt[20+35*i]==2)))
        
        test_data_img = np.array(test_data_img)
        test_trgt_img = np.array(test_trgt_img)
        test_pred_img = my_network.predict(test_data_img, verbose=1)
        
        save_img_results(test_data_img, test_trgt_img, test_pred_img, saving_dir+'test_result_img.png', n_class-1)
        
        
        # Clear memory
        train_image = None
        train_label = None
        test_image  = None
        test_label  = None
        train_dat = None
        targt_dat = None
