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
    
    
    batch_size = 32
    num_classes = 1000.0#len(listdir(directory))
    imagesNumberInDirectory = 41757
    epochs = 20
    data_augmentation = False


    save_dir = os.path.join('/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/nets/project/firstNets', 'saved_models')
    saveNow = 0

#    directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCsm/train'
#    directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCsm/test'

    #directory = '/Users/jacekkaluzny/Pictures/MAAPGX/train'
    #directorytest = '/Users/jacekkaluzny/Pictures/MAAPGX/test'
    directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/leafsPlussm/train'
    directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/leafsPlussm/test'

    #directory = '/Users/jacekkaluzny/Pictures/MAAPGX/train'
    #directorytest = '/Users/jacekkaluzny/Pictures/MAAPGX/test'
#    directory = '/Users/jacekkaluzny/Pictures/fruits/train'
#    directorytest = '/Users/jacekkaluzny/Pictures/fruits/test'

    model_name = 'leafsPlus.h5'
    model = Sequential()

    stopTraining = False
    shouldShowPlots = False


    history = None
    historyAvg = dict()


    historyAvg['mean_squared_error'] = []
    historyAvg['val_mean_squared_error'] = []
    historyAvg['mean_absolute_error'] = []
    historyAvg['val_mean_absolute_error'] = []
