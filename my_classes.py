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
from os.path import isfile, join
import json
from PIL import Image
import random
from scipy.ndimage.filters import gaussian_filter
#from scipy import misc

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=settings.batch_size, dim=(320,320,320), n_channels=3, shuffle=True):
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
#        return int(min(int(np.floor(len(self.list_IDs) / self.batch_size)),(((3+settings.saveNow)//3)**3)*40))
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
#        print(index)
        maxIndex = int(np.floor(len(self.list_IDs) / self.batch_size))
        indexes = self.indexes[(index)*self.batch_size:((index+1))*self.batch_size]
        keyList = list(self.list_IDs.keys())
        list_IDs_temp = dict()
        i = 0
        for k in indexes:
            list_IDs_temp[keyList[k]] = self.list_IDs[keyList[k]]
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    def on_epoch_end(self, epoch=int(settings.saveNow), logs=None):
        'Updates indexes after each epoch'
        os.system("caffeinate -u -t 36000 &")
#        settings.saveNow = epoch
        settings.saveNow = 0.5 + settings.saveNow
        print(settings.saveNow, epoch)
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        imagesLoaded = self.loadIMGS(list_IDs_temp)
        (x_train, y_train) = np.array(list(imagesLoaded.values())).reshape(-1,320,320,3), np.array([self.labels[x] for x in list(imagesLoaded.keys())])
#        print(list_IDs_temp)
#        print(np.array([self.labels[x] for x in list(imagesLoaded.keys())]))
        x_train = x_train.astype('float32')
        x_train /= 255.0
        
        
        return x_train, y_train


    def loadIMGS(self, paths):
        imagesL = dict()
        for name in paths:
            filename = paths[name]
            try:
                image = load_img(filename, target_size=(320, 320))
                image = img_to_array(image)
#                imageold = image
#                if(random.getrandbits(1)):
#                    if(random.getrandbits(1)):
#                        image = np.flip(image, 1)
                fil = ""
                if(random.getrandbits(1)):
                    if(random.getrandbits(1)):
    #                    print("\nin", image[50, 50, :])
                        fil+="_nb"
                        noise = np.random.normal(0, 0.5, (320, 320, 3)).astype(np.uint8).astype('float32')
                        image = np.clip(image+noise, 0.0, 255.0)
                    fil+="_bl"
#                    print("\nin", image[50:52, 50, :])
                    image = gaussian_filter(image, sigma=(random.randint(1, 2))).astype(np.uint8).astype('float32')
#                    print("out", image[50:52, 50, :])
                if(random.getrandbits(1)):
#                    print("\nin", image[50, 50, :])
                    fil+="_n"
                    noise = np.random.normal(0, 0.3, (320, 320, 3)).astype(np.uint8).astype('float32')
                    image = np.clip(image+noise, 0.0, 255.0)
#                    print("out", image[50, 50, :])
                if(random.getrandbits(1)):
                    factor = float(random.randint(950, 1300))/1000.0
                    fil+="_c"+str(factor)
#                    print("\nin", image[50:55, 50, :], "f", factor)
                    image = np.clip(128.0 + factor * image - factor * 128.0, 0.0, 255.0).astype(np.uint8).astype('float32')
#                    print("out", image[50:55, 50, :])
#                    plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)
                if(random.getrandbits(1)):
                    factor = float(random.randint(-20, 20))
                    fil+="_br"+str(factor)
#                    print("\nin", image[50:55, 50, :], "f", factor)
                    image = np.clip(image+factor, 0.0, 255.0).astype(np.uint8).astype('float32')
#                    print("out", image[50:55, 50, :])
#                    plt.imshow(image, cmap=plt.cm.gray, vmin=1, vmax=1)
#                # reshape data for the model

#                imgprev.show()
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #            if(random.getrandbits(1)):
    #                np.flip(image, 2)
                # prepare the image for the VGG model
                    
#                imgprev = Image.fromarray(image[0, :, :, :].astype(np.uint8), 'RGB')
#                imgprev.save('/Users/jacekkaluzny/Pictures/aug/a'+fil+'a'+str(random.randint(10000, 99999))+'.png')
                image = preprocess_input(image.astype('float32'))
                imagesL[name] = image
            except:
                print(filename, "doesn't exist")
            
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


