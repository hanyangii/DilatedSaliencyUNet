import keras 
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

class TestCallback(keras.callbacks.Callback):
    
    def __init__(self, test_data, test_label, save_dir, epoch=1):
        self.test_data = test_data
        self.test_label = test_label
        self.save_dir = save_dir
        self.ep = epoch
    
    def on_train_begin(self, logs={}):
        self.test_loss = []
        self.test_accuracy = []
        self.test_dicecoef = []

    def on_epoch_end(self, epoch, logs={}):
        if (epoch%self.ep)==0:
            metrics= self.model.evaluate(self.test_data, self.test_label, batch_size=16, verbose=0)
            self.test_loss.append(metrics[0])
            self.test_accuracy.append(metrics[1])
            self.test_dicecoef.append(metrics[2])
            print('Testing loss: {}, categorical-acc: {}, dice-coef: {}\n'.format(metrics[0], metrics[1], metrics[2]))
            
            if len(self.test_dicecoef) == 1 or (self.test_dicecoef[-2] < self.test_dicecoef[-1]):
                self.model.save_weights(self.save_dir+'best_train_models.h5')
                
    
    def on_train_end(self, logs={}):
        with open(self.save_dir+'test_loss.txt',"w") as n:
            for val in self.test_loss: n.write(str(val)+'\n')
         
        with open(self.save_dir+'test_categorical_accuracy.txt',"w") as n:
            for val in self.test_accuracy: n.write(str(val)+'\n')
                
        with open(self.save_dir+'test_dice_coef.txt',"w") as n:
            for val in self.test_dicecoef: n.write(str(val)+'\n')

class TrainConfig(object):
    
    def __init__(self, args):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.fold = args.fold
        self.batch_size = args.batch_size
        self.VISUALISATION = args.visualisation
        self.Patch = args.Patch
        self.hist_freq = 10 if self.VISUALISATION else 0
        self.loss = args.loss
        self.n_class = 3
        self.reduce_lr_factor = args.reduce_lr_factor
        self.reduce_lr_patience = args.reduce_lr_patience
        self.dir_name = args.dir_name
        self.interim_vis = args.interim_vis

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def print_options(arg_dict):
    print('\n======================Training Options======================')
    for idx in arg_dict:
        print(idx+': '+str(arg_dict[idx]))
    print('\n') 
    
          
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef(y_true, y_pred, ismetric=True, smooth=1e-7):
    
    label_num = y_true.shape[-1]
    if ismetric:
        y_true = tf.cast(K.argmax(y_true, axis=3), tf.float32)
        y_pred = tf.cast(K.argmax(y_pred, axis=3), tf.float32)
        y_true = tf.cast(K.greater_equal(y_true, 2), tf.float32)
        y_pred = tf.cast(K.greater_equal(y_pred, 2), tf.float32)
    else:
        y_pred = y_pred[:,:,:,-1]
        y_true = y_true[:,:,:,-1]
    
    y_true_f = tf.cast(K.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(K.flatten(y_pred), tf.float32)
        
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred, ismetric = False) + nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred)
        
        
def set_parser(parser):
    # Arguments for training
    parser.add_argument('--shuffle_epoch', type=bool,default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--bn_momentum', type=float, default=0.99)
    parser.add_argument('--TRSH', type=float, default=0.0)
    parser.add_argument('--data_chn_num', type=int, default=3)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--random_num', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', default=True)
    parser.add_argument('--no_batch_norm', dest='batch_norm',action='store_false')
    parser.add_argument('--dir_name', type=str, default='')
    parser.add_argument('--gpu_device', type=str, required=True,
                        help='Available GPU number')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--reduce_lr_factor', type=float, default=0.2)
    parser.add_argument('--reduce_lr_patience', type=int, default=4)
    parser.add_argument('--loss', type=str, default='crossentropy')
    
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--restore', dest='restore', action='store_true', default=False)
    parser.add_argument('--Patch', dest='Patch', action='store_true', default=True)
    parser.add_argument('--Slice',dest='Patch', action='store_false')
    parser.add_argument('--visualisation', dest='visualisation', action='store_true', default=False)
    parser.add_argument('--interim_vis', dest='interim_vis', action='store_true', default=False)
    
    return parser
    
    
