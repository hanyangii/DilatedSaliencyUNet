import os, time, h5py, argparse
import argparse
import nibabel as nib
import gzip

import scipy.io as sio
from scipy.ndimage import gaussian_filter
import numpy as np
import tensorflow as tf

from PIL import Image
from skimage.util.shape import view_as_windows
from skimage.transform import AffineTransform, warp
from timeit import default_timer as timer

class load_data(object):
    # Load NII data
    def __init__(self, image_name):
        # Read nifti image
        nim = nib.load(image_name)
        image = nim.get_data()
        affine = nim.affine
        self.image = image
        self.affine = affine
        self.dt = nim.header['pixdim'][4]


def data_prep(image_data):
    # Extract the 2D slices from the cardiac data
    image = image_data.image
    images = []
    for z in range(image.shape[2]):
        image_slice = image[:, :, z]
        images += [image_slice]
    images = np.array(images, dtype='float32')

    # Both theano and caffe requires input array of dimension (N, C, H, W)
    # TensorFlow (N, H, W, C)
    # Add the channel dimension, swap width and height dimension
    images = np.expand_dims(images, axis=3)

    return images

def data_prep_noSwap(image_data):
    images = image_data.image
    images = np.array(images, dtype='float32')

    return images

def data_augmentation(image):
    flip1 = np.flip(image, 1)
    flip2 = np.flip(image, 2)
    
    #Rotation
    rotation1 = np.rot90(image, k=1, axes=[0,1])
    rotation2 = np.rot90(image, k=3, axes=[0,1])
    
    augmented_data = np.concatenate([image, flip1, flip2, rotation1, rotation2], axis=2)
    #print(np.shape(augmented_data))
    return augmented_data
    
def normalisation(image, mask):
    val_img = image[np.where(mask>0)]
    mean = np.mean(val_img)
    std = np.std(val_img)
    image[np.where(mask>0)] -= mean
    image[np.where(mask>0)] /= std

    return image

def WMH_class_map(wmh_slice, icv_map, BG):
    classified_iam_map = np.zeros(wmh_slice.shape)
    if BG:
        classified_iam_map[np.where(wmh_slice==0)] = 1
        classified_iam_map[np.where(icv_map==0)] = 0
        classified_iam_map[np.where(wmh_slice>0)] = 2
    else:
        classified_iam_map[np.where(wmh_slice>0)] = 1
    
    return classified_iam_map.astype(int)


def generate_slice_data(config_dir, b_id, random_num, num_chn, TEST):
    
    print("Reading data: FLAIR")
    data_list_mri = []
    f = open('./'+config_dir+'/flair_'+str(b_id)+'.txt',"r")
    print('Open: '+'./'+config_dir+'/flair_'+str(b_id)+'.txt')
    for line in f:
        data_list_mri.append(line)
    data_list_mri = list(map(lambda s: s.strip('\n'), data_list_mri))
    
    print("Reading data: T1w")
    data_list_T1w = []
    f = open('./'+config_dir+'/T1w_'+str(b_id)+'.txt',"r")
    print('Open: '+'./'+config_dir+'/T1w_'+str(b_id)+'.txt')
    for line in f:
        data_list_T1w.append(line)
    data_list_T1w = list(map(lambda s: s.strip('\n'), data_list_T1w))

    print("Reading data: ICV")
    data_list_i = []
    f = open('./'+config_dir+'/icv_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_i.append(line)
    data_list_i = list(map(lambda s: s.strip('\n'), data_list_i))
    
    print("Reading data: CSF")
    data_list_c = []
    f = open('./'+config_dir+'/csf_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_c.append(line)
    data_list_c = list(map(lambda s: s.strip('\n'), data_list_c))

    
    print("Reading data: LABELS")
    data_list_l = []
    f = open('./'+config_dir+'/label_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_l.append(line)
    data_list_l = list(map(lambda s: s.strip('\n'), data_list_l))
    
    print("Reading data: IAM")
    data_list_a = []
    f = open('./'+config_dir+'/iam_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_a.append(line)
    data_list_a = list(map(lambda s: s.strip('\n'), data_list_a))

    id = 0
    test_data = []
    test_trgt = []
    for data in data_list_mri:
        if os.path.isfile(data) :
            loaded_data_f = load_data(data)
            loaded_data_T1w = load_data(data_list_T1w[id])
            loaded_data_i = load_data(data_list_i[id])
            loaded_data_l = load_data(data_list_l[id])
            loaded_data_c = load_data(data_list_c[id])
            loaded_data_a = load_data(data_list_a[id])

            loaded_image_f = data_prep(loaded_data_f)
            loaded_image_T1w = data_prep(loaded_data_T1w)
            loaded_image_i = data_prep(loaded_data_i)
            loaded_image_l = data_prep(loaded_data_l)
            loaded_image_c = data_prep(loaded_data_c)
            loaded_image_a = data_prep(loaded_data_a)

            loaded_image_f = np.squeeze(loaded_image_f)
            loaded_image_T1w = np.squeeze(loaded_image_T1w)
            loaded_image_i = np.squeeze(loaded_image_i)
            loaded_image_l = np.squeeze(loaded_image_l)
            loaded_image_c = np.squeeze(loaded_image_c)
            loaded_image_a = np.squeeze(loaded_image_a)
            
            loaded_image_c = ~loaded_image_c.astype(bool)
            loaded_image_i = np.multiply(loaded_image_i, loaded_image_c.astype(int))
            
            val_dat = np.multiply(loaded_image_f, loaded_image_i)
            val_T1w = np.multiply(loaded_image_T1w, loaded_image_i)
            val_iam = np.multiply(loaded_image_a, loaded_image_i)
            val_lbl = np.multiply(loaded_image_l, loaded_image_i)
            
            
            loaded_val_mri = normalisation(val_dat, loaded_image_i)
            loaded_val_T1w = normalisation(val_T1w, loaded_image_i)
            loaded_val_iam = val_iam
           
            val_lbl = WMH_class_map(val_lbl, loaded_image_i, True)
            
            if id == 0:
                test_data = loaded_val_mri
                test_T1w = loaded_val_T1w
                test_iam = loaded_val_iam
                test_trgt = val_lbl
            else:
                test_data = np.concatenate((test_data,loaded_val_mri), axis=0)
                test_T1w = np.concatenate((test_T1w,loaded_val_T1w), axis=0)
                test_iam = np.concatenate((test_iam,loaded_val_iam), axis=0)
                test_trgt = np.concatenate((test_trgt,val_lbl), axis=0)
                print("test_FLAIR_data SHAPE: " + str(test_data.shape) + " | " + str(id))
            if TEST and len(test_data)>100: break
            id += 1
    
    va, vb, vc = test_data.shape
    test_data = np.reshape(test_data, (va, vb, vc, 1))
    if num_chn == 2:
        test_iam  = np.reshape(test_iam,  (va, vb, vc, 1))
        test_data = np.concatenate((test_data, test_iam), axis=3)
    if num_chn == 3:
        test_T1w  = np.reshape(test_T1w,  (va, vb, vc, 1))
        test_iam  = np.reshape(test_iam,  (va, vb, vc, 1))
        test_data = np.concatenate((test_data, test_iam, test_T1w), axis=3)
    test_trgt = np.reshape(test_trgt, (va, vb, vc, 1))
    
    return test_data, test_trgt, data_list_l

def generate_patch_data(config_dir, b_id, TRSH, win_shape, random_num, num_chn, TEST):
    # LOAD TRAINING DATA
    print("Reading data: FLAIR")
    data_list_mri = []
    f = open('./'+config_dir+'/flair_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_mri.append(line)
    data_list_mri = list(map(lambda s: s.strip('\n'), data_list_mri))
    
    print("Reading data: T1W")
    data_list_T1w = []
    f = open('./'+config_dir+'/T1w_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_T1w.append(line)
    data_list_T1w = list(map(lambda s: s.strip('\n'), data_list_T1w))

    print("Reading data: ICV")
    data_list_icv = []
    f = open('./'+config_dir+'/icv_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_icv.append(line)
    data_list_icv = list(map(lambda s: s.strip('\n'), data_list_icv))
    
    print("Reading data: CSF")
    data_list_csf = []
    f = open('./'+config_dir+'/csf_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_csf.append(line)
    data_list_csf = list(map(lambda s: s.strip('\n'), data_list_csf))

    #WMH label file
    print("Reading data: WMH")
    data_list_lbl = []
    f = open('./'+config_dir+'/label_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_lbl.append(line)
    data_list_lbl = list(map(lambda s: s.strip('\n'), data_list_lbl))
    
    print("Reading data: IAM")
    data_list_iam = []
    f = open('./'+config_dir+'/iam_'+str(b_id)+'.txt',"r")
    for line in f:
        data_list_iam.append(line)
    data_list_iam = list(map(lambda s: s.strip('\n'), data_list_iam))
    
    id = 0
    train_data = []
    train_trgt = []
    for data in data_list_mri:
        if os.path.isfile(data):
            #print(data)

            loaded_data_mri = load_data(data)
            loaded_data_T1w = load_data(data_list_T1w[id])
            loaded_data_icv = load_data(data_list_icv[id])
            loaded_data_csf = load_data(data_list_csf[id])
            loaded_data_lbl = load_data(data_list_lbl[id])
            loaded_data_iam = load_data(data_list_iam[id])

            #Load Images as float array 

            loaded_image_mri = data_prep_noSwap(loaded_data_mri)
            loaded_image_T1w = data_prep_noSwap(loaded_data_T1w)
            loaded_image_icv = data_prep_noSwap(loaded_data_icv)
            loaded_image_csf = data_prep_noSwap(loaded_data_csf)
            loaded_image_lbl = data_prep_noSwap(loaded_data_lbl)
            loaded_image_iam = data_prep_noSwap(loaded_data_iam)

            loaded_image_mri = np.squeeze(loaded_image_mri)
            loaded_image_T1w = np.squeeze(loaded_image_T1w)
            loaded_image_iam = np.squeeze(loaded_image_iam)
            loaded_image_icv = np.squeeze(loaded_image_icv)
            loaded_image_csf = np.squeeze(loaded_image_csf)
            loaded_image_lbl = np.squeeze(loaded_image_lbl)
            
            loaded_image_csf = ~loaded_image_csf.astype(bool)
            loaded_image_icv = np.multiply(loaded_image_icv, loaded_image_csf)
            
            loaded_image_mri = np.multiply(loaded_image_mri, loaded_image_icv)
            loaded_image_iam = np.multiply(loaded_image_iam, loaded_image_icv)
            loaded_image_T1w = np.multiply(loaded_image_T1w, loaded_image_icv)
            loaded_image_lbl = np.multiply(loaded_image_lbl, loaded_image_icv)

            loaded_image_mri = normalisation(loaded_image_mri, loaded_image_icv)
            loaded_image_T1w = normalisation(loaded_image_T1w, loaded_image_icv)
            
            # label processing
            loaded_image_lbl = WMH_class_map(loaded_image_lbl, loaded_image_icv, True)
            
            print("Number of WMH voxels: " + str(np.sum(loaded_image_lbl)))
            
            if not win_shape[0]==256:
                not_yet_labelled = np.argwhere(loaded_image_lbl > TRSH) #default TRSH = 0
                not_yet_labelled_rand_perm_id = np.random.permutation(len(not_yet_labelled))
                not_yet_labelled_rand_perm = not_yet_labelled[not_yet_labelled_rand_perm_id[:random_num]]
                not_yet_labelled = not_yet_labelled_rand_perm
                print("Number of samples: " + str(len(not_yet_labelled)))

                pad_shape = (int(np.floor(win_shape[0]/2)),int(np.floor(win_shape[1]/2)),int(np.floor(win_shape[2]/2)))

                #Zero padding along each axis
                mra_data_padded = np.pad(loaded_image_mri, 
                                            ((pad_shape[0],pad_shape[0]),
                                            (pad_shape[1],pad_shape[1]),
                                            (pad_shape[2],pad_shape[2])), 
                                            'constant', constant_values=0)
                block_mra_data = view_as_windows(mra_data_padded, win_shape)

                T1w_data_padded = np.pad(loaded_image_T1w, 
                                            ((pad_shape[0],pad_shape[0]),
                                            (pad_shape[1],pad_shape[1]),
                                            (pad_shape[2],pad_shape[2])), 
                                            'constant', constant_values=0)
                block_T1w_data = view_as_windows(T1w_data_padded, win_shape)

                iam_data_padded = np.pad(loaded_image_iam, 
                                            ((pad_shape[0],pad_shape[0]),
                                            (pad_shape[1],pad_shape[1]),
                                            (pad_shape[2],pad_shape[2])), 
                                            'constant', constant_values=0)
                block_iam_data = view_as_windows(iam_data_padded, win_shape)

                vessel_data_padded = np.pad(loaded_image_icv, 
                                            ((pad_shape[0],pad_shape[0]),
                                                (pad_shape[1],pad_shape[1]),
                                                (pad_shape[2],pad_shape[2])),
                                                'constant', constant_values=0)
                block_vessel_data = view_as_windows(vessel_data_padded, win_shape)

                vessel_labelled_padded = np.pad(loaded_image_lbl, 
                                                ((pad_shape[0],pad_shape[0]),
                                                    (pad_shape[1],pad_shape[1]),
                                                    (pad_shape[2],pad_shape[2])),
                                                    'constant', constant_values=0)
                block_vessel_labelled = view_as_windows(vessel_labelled_padded, win_shape)

                num_augment = 1
                train_dat = np.zeros((len(not_yet_labelled)*num_augment,win_shape[0],win_shape[1],num_chn))

                targt_dat = np.zeros((len(not_yet_labelled)*num_augment,win_shape[0],win_shape[1],1))

                ''' Random Patch Extraction ''' 
                idx = 0
                for indices in not_yet_labelled:
                    for ii_aug in range(num_augment):

                        rand_idx = np.random.randint((pad_shape[0]-5)*-1,pad_shape[0]-5,size=(2,))
                        [x,y,z] = indices
                        x, y, z = x+rand_idx[0],y+rand_idx[1],z
                        
                        if (x > block_mra_data.shape[0]-1) or (y>block_mra_data.shape[1]-1):
                            continue

                        chn0 = block_mra_data[x,y,z]
                        chn1 = block_vessel_data[x,y,z]
                        chn2 = block_T1w_data[x,y,z]
                        chn3 = block_iam_data[x,y,z]
                        lbls = block_vessel_labelled[x,y,z]


                        train_dat[idx,:,:,0] = chn0[:,:,0]
                        if num_chn == 2:
                            train_dat[idx,:,:,1] = chn3[:,:,0]
                        elif num_chn == 3:
                            train_dat[idx,:,:,1] = chn3[:,:,0]
                            train_dat[idx,:,:,2] = chn2[:,:,0]
                        elif num_chn == 4:
                            train_dat[idx,:,:,1] = chn3[:,:,0]
                            train_dat[idx,:,:,2] = chn2[:,:,0]
                            train_dat[idx,:,:,3] = chn1[:,:,0]
                        targt_dat[idx,:,:,0] =lbls[:,:,0]
                        idx += 1
            else: 
                loaded_image_mri = np.expand_dims(data_augmentation(loaded_image_mri), axis=3)
                loaded_image_iam = np.expand_dims(data_augmentation(loaded_image_iam), axis=3)
                loaded_image_T1w = np.expand_dims(data_augmentation(loaded_image_T1w), axis=3)
                loaded_image_mri = np.transpose(loaded_image_mri, axes=(2,0,1,3))
                loaded_image_iam = np.transpose(loaded_image_iam, axes=(2,0,1,3))
                loaded_image_T1w = np.transpose(loaded_image_T1w, axes=(2,0,1,3))
                train_dat = np.concatenate((loaded_image_mri, loaded_image_iam, loaded_image_T1w), axis=3)
                loaded_image_lbl = np.expand_dims(data_augmentation(loaded_image_lbl),axis=3)
                targt_dat = np.transpose(loaded_image_lbl, axes=(2,0,1,3))
            
            
            if id == 0:
                train_data = train_dat
                train_trgt = targt_dat
            else:
                train_data = np.concatenate((train_data,train_dat), axis=0)
                train_trgt = np.concatenate((train_trgt,targt_dat), axis=0)
                print("train_data SHAPE: " + str(train_data.shape) + " | " + str(id))
            if TEST and len(train_data)>100: break
            id += 1

    return train_data, train_trgt, data_list_lbl 


