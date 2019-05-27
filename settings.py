#!/usr/bin/env python3
from keras.models import Sequential
import os
from os import listdir
from os.path import isfile, join

def init():
    global batch_size
    global removeEveryNBatch
    global num_classes
    global imagesNumberInDirectory
    global epochs
    global data_augmentation
    global num_predictions
    global save_dir
    global saveNow
    global directory
    global directorytrain
    global directorytest
    global directoryval
    global model_name
    global model
    global stopTraining
    global shouldShowPlots
    global history
    global historyAvg
    global labelnr
    
    labelnr = dict()
    batch_size = 16
    num_classes = 54#len(listdir(directory))
    epochs = 30
    data_augmentation = False


    save_dir = os.path.join('/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/nets/project/firstNets', 'saved_models')
    saveNow = 0

    directory = '/Users/jacekkaluzny/Pictures/4paramsmay/data'
    directorytrain = directory+'/train'
    directorytest = directory+'/test'
    directoryval = directory+'/val'
    

    model_name = '4paramitertsMayR.h5'
    model = Sequential()

    stopTraining = False
    shouldShowPlots = False


    history = None
    historyAvg = dict()
    historyAvg['mean_squared_error'] = []
    historyAvg['val_mean_squared_error'] = []
    historyAvg['mean_absolute_error'] = []
    historyAvg['val_mean_absolute_error'] = []
    def showPlots():
        historyAvg = self.historyAvg
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
