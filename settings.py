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
    global directorytest
    global model_name
    global model
    global stopTraining
    global shouldShowPlots
    global history
    global historyAvg
    global labelnr
    
    labelnr = dict()
    batch_size = 16
    num_classes = 9#len(listdir(directory))
    epochs = 5
    data_augmentation = False


    save_dir = os.path.join('/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/nets/project/firstNets', 'saved_models')
    saveNow = 0

#    directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCsm/train'
#    directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCsm/test'

    #directory = '/Users/jacekkaluzny/Pictures/MAAPGX/train'
    #directorytest = '/Users/jacekkaluzny/Pictures/MAAPGX/test'
    directory = '/Volumes/M/magisterka/classes8/data/train'
    directorytest = '/Volumes/M/magisterka/classes8/data/test'

    #directory = '/Users/jacekkaluzny/Pictures/MAAPGX/train'
    #directorytest = '/Users/jacekkaluzny/Pictures/MAAPGX/test'
#    directory = '/Users/jacekkaluzny/Pictures/fruits/train'
#    directorytest = '/Users/jacekkaluzny/Pictures/fruits/test'

    model_name = 'regresion8points.h5'
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
