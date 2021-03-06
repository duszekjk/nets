#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import settings
settings.init()


import keras
import keras.applications as kapp


import six
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, serialize
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras import regularizers
from keras import losses
from keras.layers.advanced_activations import LeakyReLU

from os import listdir
from os.path import isfile, join
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from my_classes import DataGenerator

import random
import threading
from itertools import islice
import matplotlib.pyplot as plt
from operator import add
import requests

from PIL import Image
#import keyboard  # using module keyboard
#from getkey import getkey, keys

labelsb = dict()
labels = dict()
labelsMin = dict()
labelsMax = dict()
labelsAvg = dict()

imagesCombiner = dict()
imagesBlocker = 0

imagesCombinerLoad = dict()
imagesBlockerLoad = 0

j=0


os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.40.1/bin'

def showPlots():
    global historyAvg
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


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

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

def loadIMGS(paths):
    imagesL = dict()
    for name in paths:
        filename = paths[name]
        image = load_img(filename, target_size=(320, 320))
        image = img_to_array(image)
#            if(random.getrandbits(1)):
#                image = np.flip(image, 1)
#            if(random.getrandbits(1)):
#                image = gaussian_filter(image, sigma=(3))
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#            if(random.getrandbits(1)):
#                np.flip(image, 2)
        # prepare the image for the VGG model
        image = preprocess_input(image)
        imagesL[name] = image
    return imagesL
def loadThisPhotoNames(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    global j
#    j = 0
    images = dict()
            # load an image from file
    photoData = className.split(", ")
    filename = directory + '/' + photoData[0] + ".jpg"

    # load an image from file
    label = np.array(className.split(","))[1:].astype(float)
#    label[0] = (label[0]*5.0)-2.5  #ac
#    label[1] = (label[1]*2.0)-1.0
#    label[2] = (label[2]/3.0)
#    label[3] = (label[3]/102.0)-1.0 #max vigor
#    label[4] = (label[4]*3.34)-2.172 #determinism
    label = [label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9], label[10], label[11], label[12], label[13], label[17], label[24], label[25], label[26], label[27], label[28], label[29], label[30], label[31], label[33], label[-1]]
#    print(label)
    for iii in range(0, len(label)):
        if str(label[iii]) == "nan":
            print("usunieto", label)
            return
        try:
            if labelsMin[iii] > label[iii]:
                labelsMin[iii] = label[iii]
        except:
            labelsMin[iii] = label[iii]
        try:
            if labelsMax[iii] < label[iii]:
                labelsMax[iii] = label[iii]
        except:
            labelsMax[iii] = label[iii]
        try:
            labelsAvg[iii] = labelsAvg*(iiii)+label[iii]/(iiii+1)
        except:
            labelsAvg[iii] = label[iii]
            
    
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
    filename = directory + '/' + photoData[0] + "_with_bg_a.jpg"
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
    filename = directory + '/' + photoData[0] + "_with_bg_b.jpg"
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
    filename = directory + '/' + photoData[0] + "_with_bg_c.jpg"
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
    filename = directory + '/' + photoData[0] + "_with_bg_d.jpg"
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
    filename = directory + '/' + photoData[0] + "_with_bg_e.jpg"
    if (isfile(filename)):
        labels[j] = label
        images[j] = filename
        j += 1
#    labels[j] = label#[label[0], label[1], label[4]]
#    images[j] = filename
#    print(filename, label)
#    j += 1
    imagesCombiner.update(images)
    imagesBlocker += 1
#    print("class end: \t", className, "\t",len(images))

def loadPhotosNamesInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    images = dict()
#    j = 0
    #    imagesCombiner = dict()
    #    imagesBlocker = 0
#    print("cl ", className)
    for name in listdir(directory + "/" + className):
        if name[0] != '.':
            # load an image from file
            filename = directory + '/' + className + '/' + name
            label = np.array(className.split("+")).astype(float)
#            label[0] = ((label[0]-0.25)*2.67)-1.0   #ac
#            label[1] = (label[1]*2.0)-1.0
#            label[2] = (label[2]/3.0)
#            label[3] = (label[3]/100.0)-1.0 #max vigor
#            label[4] = ((label[4]-0.3)*3.08)-1.54 #determinism
#            print(label)
#            label =  className#np.array([((classNameInt+30)%100)/100.0, (classNameInt//1000)/100.0]).astype(float)
            #            label = (label*2.0) - 1.0
#            labelsb[settings.labelnr[className]] =  className #(((classNameInt//1000000)/1000.0)*2) - 1.0
            labels[i*100000+j] = label#[label[0], label[1], label[4]]
            images[i*100000+j] = filename
            j += 1
    imagesCombiner.update(images)
#    print(len(imagesCombiner))
    imagesBlocker += 1



def load_photos(directory, names = False, csv=""):
    global imagesBlocker
    global imagesCombiner
    
#    imagesCombiner.clear()
    threads = []
    images = dict()
    i = 0
    
    classesNames = []
    if csv == "":
        classesNames = listdir(directory)
    else:
        classesNames = open(csv, "r")
    for className in classesNames:
        if className[0] != '.':
            #            print("class: ", className, i, names)
            if csv != "":
                imagesBlocker = 0
#                threads.append(threading.Thread(target=loadThisPhoto, args=(directory, className, i)))
                loadThisPhotoNames(directory, className, i)
            else:
                #                threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
#                print(className)
                if names == True:
                    loadPhotosNamesInCategory(directory, className, i)
                else:
                    threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
        i += 1
    if names != True:
        print("class state: \tname\tnr of images (", i, ")")
        for thread in threads:
            thread.start()
            
            while imagesBlocker != i:
                time.sleep(2.0)
                print(imagesBlocker, i)
        for thread in threads:
            i -= 1
            thread.join()
    images = imagesCombiner
    #    print(list(imagesCombiner.keys()))
    print("All loaded")
    keys =  list(imagesCombiner.keys())
    random.shuffle(keys)
    for key in keys:
        images[key] = imagesCombiner[key]
    #    print(list(images.keys()))
    return images


analisedData = None




imagesTest = load_photos(settings.directoryval, names = True, csv = settings.directoryval+"/classes.csv").copy()
imagesCombiner.clear()
imagesTrain = load_photos(settings.directorytrain, names = True, csv = settings.directorytrain+"/classes.csv").copy()
imagesCombiner.clear()
#(imagesTrain, imagesTest) = chunks(images, int(len(images)*995/1000))
imagesChunks =  createBatch(imagesTrain, 4*(int(settings.batch_size*1.25))+1)

print(labelsMin[0], labelsMax[0])

for jjj in range(0, len(labelsMin)):
    if labelsMin[jjj] == labelsMax[jjj]:
        labelsMin[jjj] -= 1.0
        labelsMax[jjj] += 1.0001
for iii in range(0, len(labels)):
    for jjj in range(0, len(labels[iii])):
        labels[iii][jjj] = ((labels[iii][jjj] - labelsMin[jjj]) / (labelsMax[jjj] - labelsMin[jjj]))*2.0 - 1.0

#print(len(labels[0]), len(labelsMax), len(labelsMin))
#print("max: ",labelsMax.values())
#print("min: ",labelsMin.values())

# 0.7
# 0.3
# 1.0
# 0.57
# 0.14

numberOfParameters = len(labels[0])
print(labels[0])

print('Loaded Images: %d / %d' % (int(len(imagesTrain)), int(len(imagesTest))), '\tparameters: ', numberOfParameters)
#print(labels[0])

# Generators
#print(imagesTest)
training_generator = DataGenerator(imagesTrain, labels)
validation_generator = DataGenerator(imagesTest, labels)
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
model_path = os.path.join(settings.save_dir, settings.model_name)


epoch_start = 0
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
    
    
# ----------------------------------------- Model start ----------------------------------------#
settings.model_name += "R"
model_path = os.path.join(settings.save_dir, settings.model_name)
    
settings.model = Sequential()
settings.model.add(Conv2D(32, (3, 3), padding='same',input_shape=(320, 320, 3), name='conv2d_1'))
#kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
settings.model.add(LeakyReLU(alpha=0.001, name='leaky_re_lu_1'))
settings.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))

settings.model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_2'))
settings.model.add(LeakyReLU(alpha=0.001, name='leaky_re_lu_2'))
settings.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
#    settings.model.add(Dropout(0.001))

settings.model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_4'))
settings.model.add(LeakyReLU(alpha=0.001, name='leaky_re_lu_4'))
settings.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3'))
#settings.model.add(Dropout(0.3))
#
settings.model.add(Conv2D(64, (3, 3), padding='same', name='cconv2d_6'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_6'))
settings.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4'))
#

settings.model.add(Flatten(name='flatten_1'))
settings.model.add(Dense(320, name='dense_1a'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_12'))
#
settings.model.add(Dropout(0.4, name='dropout_last'))

settings.model.add(Dense(320, name='dense_1b'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_12b'))
settings.model.add(Dropout(0.4))

settings.model.add(Dense(settings.num_classes*numberOfParameters, name='dense_1'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_13'))


settings.model.add(Dense(numberOfParameters, name='dense_2'))
#                             , kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
settings.model.add(Activation('linear')) 


settings.model.summary()


# initiate RMSprop optimizer

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = 'adam'#keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


print("1 train\n2 test")
traintestmode = input()

if "1" not in traintestmode:
#settings.model.load_weights(settings.save_dir+"/"+settings.model_name)
#    settings.model.load_weights(settings.save_dir+"/"+"weights-improvement-maskR-43-0.00785.hdf5")
#print(os.system("ls -al \""+settings.save_dir+"\""))
    settings.model.load_weights(settings.save_dir+"/load.hdf5")
#    settings.model.load_weights(settings.save_dir+"/12treesR")
    print("model loaded")


#settings.model = multi_gpu_model(settings.model, gpus=2)
settings.model.compile(loss='mean_squared_error',
                     optimizer=opt,
                     metrics=['categorical_accuracy', 'mean_squared_error', 'mean_absolute_error', 'accuracy'])
filepath=settings.save_dir+"/weights-improvement-"+settings.model_name+"-{epoch:02d}-{mean_absolute_error:.5f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
webpage = RemoteMonitor(root='http://trees.duszekjk.com', path='/liveupdates/')
callbacks_list = [checkpoint, webpage]
historyAvg = []

#steps_per_epoch=1200,
#print("1 train\n2 test")
#traintestmode = input()
if "2" not in traintestmode:
    History = settings.model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          use_multiprocessing=False,
                                          workers=1, epochs=settings.epochs, verbose = 1, callbacks=callbacks_list, initial_epoch = epoch_start, max_queue_size=100)

    #
    settings.model.save(model_path)
    stopTraining = True
else:


    stopTraining = True



    print("tests:")
    imagesCombiner.clear()
    j=0
    labels.clear()

    j_old = j
    listOfTests = load_photos(settings.directorytest, names = True, csv = settings.directorytest+"/classes.csv").copy()
    print('Loaded Images Test: %d' % int(len(listOfTests)))
    myTest = loadIMGS(listOfTests)
    print('Loaded Images Test: %d' % int(len(myTest)))
    for iii in range(0, len(labels)):
        for jjj in range(0, len(labels[iii])):
            labels[iii][jjj] = ((labels[iii][jjj] - labelsMin[jjj]) / (labelsMax[jjj] - labelsMin[jjj])) * 2.0 - 1.0
    #print(labels)
    (my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,320,320,3), np.array([labels[x+j_old] for x in list(labels)])
    my_x_test = my_x_test.astype('float32')
    my_x_test /= 255.0



    scores = settings.model.evaluate(my_x_test[:-13], my_y_test[:-13], verbose=1)
    if(len(scores)<3):
        scores = settings.model.evaluate(my_x_test[:-1], my_y_test[:-1], verbose=1)
    print('Test '+settings.model.metrics_names[0]+':', scores[0])
    print('Test '+settings.model.metrics_names[2]+':', scores[2])
    print('Test '+settings.model.metrics_names[3]+':', scores[3])

    classes = settings.model.predict(my_x_test, batch_size=16)
    j = 0

    #print(classes)
    if len(classes[0]) == numberOfParameters:
        errors = []
        arrayR = []
        for param in range(0,35):
            arrayR.append("[")
        arrayS = "["
        jjjj = 0
        for classesProbs in classes:
    #        if jjjj == 0 or jjjj == 15 or jjjj == 29 or jjjj == 34:
            iiii = 0
            iiiib = 0
            arrayS += "\""
            for param in classesProbs:
                if iiiib == 14:
                    arrayR[iiiib] += "4.0, "
                    iiiib += 1
                while iiiib == 15 or iiiib == 16:
                    arrayR[iiiib] += "0.1, "
                
                    iiiib += 1
                while (iiiib >= 18 and iiiib <= 23) or iiiib == 32:
                    arrayR[iiiib] += "1.0, "
                
                    iiiib += 1
    #            errors.append((param - labels[jjjj][iiii])**2)
    #            print((param - labels[jjjj][iiii])**2, ", ")
                arrayR[iiiib] += str(round((labelsMax[iiii]-labelsMin[iiii])*((1.0 + min(max(param, -1.0),1.0))/2.0)+labelsMin[iiii], 6))+", "
    #            aaa = (param - labels[jjjj][iiii])**2 #((round((labelsMax[iiii]-labelsMin[iiii])*((1.0 + min(max(param, -1.0),1.0))/2.0)+labelsMin[iiii], 6) - round((labelsMax[iiii]-labelsMin[iiii])*((1.0 + labels[jjjj][iiii])/2.0)+labelsMin[iiii], 6)))**2
    #                if aaa > 0.011:
    #                    print(jjjj)
    #            errors.append(((round((labelsMax[iiii]-labelsMin[iiii])*((1.0 + min(max(param, -1.0),1.0))/2.0)+labelsMin[iiii], 6) - round((labelsMax[iiii]-labelsMin[iiii])*((1.0 + labels[jjjj][iiii])/2.0)+labelsMin[iiii], 6)))**2)
                iiii += 1
                iiiib += 1
        #        for description in my_y_test[jjjj]:
                 
            arrayS += str(listOfTests[jjjj].split("/")[-1])+"+"
            arrayS = arrayS[:-1] + "\", "
            jjjj += 1
        
        iiii = 0
        for param in range(0,35):
            arrayR[iiii] += "],"
            iiii += 1
        arrayS += "]"
        iiii = 0
        for tree in arrayR:
            print(tree)
            iiii += 1
        print(arrayS)
        errors.sort()
    #    print(errors)
        

    else:
        for classesProbs in classes:
            print(labelsb[np.argmax(my_y_test[j])], labelsb[np.argmax(classesProbs)])
            j += 1


    #    print(prediction)
