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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=settings.batch_size, dim=(512,512,512), n_channels=3, shuffle=True):
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
#         Find list of IDs
#        print(indexes)
        keyList = list(self.list_IDs.keys())
        list_IDs_temp = dict()
        i = 0
        for k in indexes:
            list_IDs_temp[keyList[k]] = self.list_IDs[keyList[k]]
#            print(list_IDs_temp[keyList[k]], self.labels[keyList[k]] , end=", ")
#        print("\n--------------------------")
#        list_IDs_temp = [keyList[k]:self.list_IDs[keyList[k]] for k in indexes]
#        print([list_IDs_temp.values(), self.labels[list_IDs_temp.keys()]])
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    def on_epoch_end(self, epoch=int(settings.saveNow), logs=None):
        if(logs != None):
            settings.historyAvg['mean_squared_error'] += logs['mean_squared_error']
            settings.historyAvg['val_mean_squared_error'] += logs['val_mean_squared_error']
            settings.historyAvg['mean_absolute_error'] += logs['mean_absolute_error']
            settings.historyAvg['val_mean_absolute_error'] += logs['val_mean_absolute_error']
            model_path = os.path.join(settings.save_dir, settings.model_name)
            with open(model_path+".json", 'w') as fp:
                json.dump(settings.historyAvg, fp)
            if(settings.shouldShowPlots == True):
                showPlots(settings.historyAvg)
                settings.shouldShowPlots = False
        'Updates indexes after each epoch'
        os.system("caffeinate -u -t 36000 &")
#        settings.saveNow = epoch
        settings.saveNow = 0.5 + settings.saveNow
        print(settings.saveNow, epoch)
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

#        if settings.saveNow > 1:
#            model_path = os.path.join(settings.save_dir, settings.model_name)
#            settings.model.save(model_path)
#            print("saved")
#        else:
#            if not os.path.isdir(settings.save_dir):
#                os.makedirs(settings.save_dir)




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
#        where_are_NaNs = np.isnan(x_train)
#        x_train[where_are_NaNs] = 0.0
#        where_are_NaNs = np.isnan(y_train)
#        y_train[where_are_NaNs] = 0.0
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
#            print(filename, self.labels[name])
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


def showPlots(historyAvg):
    print("plots:")
    plt.plot(list( map(add, historyAvg['mean_squared_error'][3:], historyAvg['val_mean_squared_error'][3:])))
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train + test'], loc='upper left')
    plt.show()
    plt.plot(list( map(add, historyAvg['mean_absolute_error'][3:], historyAvg['val_mean_absolute_error'][3:])))
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train + test'], loc='upper left')
    plt.show()
    
    plt.plot(historyAvg['mean_squared_error'][3:])
    plt.plot(historyAvg['val_mean_squared_error'][3:])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(historyAvg['mean_absolute_error'][3:])
    plt.plot(historyAvg['val_mean_absolute_error'][3:])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history['mean_squared_error'][3:])
    plt.plot(history['val_mean_squared_error'][3:])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('batches')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history['mean_absolute_error'][3:])
    plt.plot(history['val_mean_absolute_error'][3:])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('batches')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
