import numpy as np
import keras
from os import listdir
from os.path import isfile, join
import os

import settings

import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
import keras.applications as kapp
from keras.models import load_model
import json

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=128, dim=(512,512,512), n_channels=3, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
#        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
#         Find list of IDs
#        print(indexes)
        keyList = list(self.list_IDs.keys())
        list_IDs_temp = dict()
        i = 0
        for k in indexes:
            list_IDs_temp[keyList[k]] = self.list_IDs[keyList[k]]
            if i > (((settings.saveNow+10)//10)**3)*10:
                break
            i += 1
#        list_IDs_temp = [keyList[k]:self.list_IDs[keyList[k]] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        os.system("caffeinate -u -t 36000 &")
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        if settings.saveNow > 1:
            model_path = os.path.join(settings.save_dir, settings.model_name)
            settings.model.save(model_path)
            print("saved")
        else:
            if not os.path.isdir(settings.save_dir):
                os.makedirs(settings.save_dir)
        settings.saveNow += 1




    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        #        imagesTrain
        #        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #        y = np.empty((self.batch_size), dtype=int)
#        print(list_IDs_temp)
        imagesLoaded = self.loadIMGS(list_IDs_temp)
        (x_train, y_train) = np.array(list(imagesLoaded.values())).reshape(-1,512,512,3), np.array([self.labels[x] for x in list(imagesLoaded.keys())])
        x_train = x_train.astype('float32')
        x_train /= 255.0
        # Generate data
#        for i, ID in enumerate(list_IDs_temp):
#            # Store sample
#            X[i,] = np.load('data/' + ID + '.npy')
#            
#            # Store class
#            y[i] = self.labels[ID]
        
        return x_train, y_train


    def loadIMGS(self, paths):
        imagesL = dict()
        for name in paths:
            filename = paths[name]
            image = load_img(filename, target_size=(512, 512))
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            imagesL[name] = image
        return imagesL
    
    def createBatch(data, SIZE=1000):
        endList = list()
        smallDict = dict()
        i = 0
        for key in data:
            #        print("key:", key)
            if i%SIZE != SIZE-1:
                smallDict[key] = data[key]
            else:
                endList.append(smallDict)
                smallDict = dict()
            i+=1
        return endList
