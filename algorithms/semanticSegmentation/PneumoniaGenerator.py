import os
import csv
import random
import pydicom
import numpy as np
import cv2
import pandas as pd
from skimage import measure
from skimage.transform import resize
import scipy
import tensorflow as tf
from tensorflow import keras
import random
from matplotlib import pyplot as plt



class PneumoniaGenerator(keras.utils.Sequence):
    '''
    folder: folder with the train/test images
    filenames: train/test filenames
    '''
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=320, shuffle=True, augment=False, predict=False):
        self.folder = folder # folder with the train/test images
        self.filenames = filenames # train/test filenames
        self.pneumonia_locations = pneumonia_locations # dictionary with patient ids and their locations {'idpatient': [[location], [location],..]}
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    '''
    given a filename, loads the image and creates the binarymask of pneumonia (returns both of them)
    '''
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1

        # resize both image and mask
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)


        # if augment then horizontal flip half the time
        if self.augment:
            s1, s2 = img.shape 

            # 50% of flip
            if random.random() > 0.5:
                img = np.fliplr(img)
                msk = np.fliplr(msk)

            # 50% of shift
            if random.random() > 0.5:
                shift_1 = random.randint(-15, 15)
                shift_2 = random.randint(-15, 15)
                img = scipy.ndimage.shift(img, (shift_1,shift_2), mode='reflect')
                msk = scipy.ndimage.shift(msk, (shift_1,shift_2), mode='reflect')

            # 50% of rotate
            if random.random() > 0.5:
                rotation = random.randint(-10, 10)
                img = scipy.ndimage.rotate(img, rotation, axes=(1, 0), mode='reflect')
                msk = scipy.ndimage.rotate(msk, rotation, axes=(1, 0), mode='reflect')

            s11, s22 = img.shape 

            if s1!= s11:
                # the image has changed its size, crop it
                starty = int((s11-s1)/2)
                startx = int((s22-s2)/2)
                img = img[starty:starty+s1,startx:startx+s2]
                msk = msk[starty:starty+s1,startx:startx+s2]
         


        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        
        return img, msk
    

    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


'''
add iamge augmentation: rotations
'''