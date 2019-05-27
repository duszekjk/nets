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

from PIL import Image
#import keyboard  # using module keyboard
from getkey import getkey, keys

labelsb = dict()
labels = dict()

imagesCombiner = dict()
imagesBlocker = 0

imagesCombinerLoad = dict()
imagesBlockerLoad = 0

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
    illSum = max(int(len(paths)//100), 1)
#    print(illSum, int(len(paths)//100))
    ill = 0
    for name in paths:
        ill += 1
        if ill%illSum != 1:
            continue
        filename = paths[name]
        image = load_img(filename, target_size=(512, 512))
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        imagesL[name] = image
    return imagesL

def loadPhotosNamesInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    images = dict()
    j = 0
    #    imagesCombiner = dict()
    #    imagesBlocker = 0
    for name in listdir(directory + "/" + className):
        if name[0] != '.':
            # load an image from file
            filename = directory + '/' + className + '/' + name
#            classNameInt = int(className)
#            label = np.array([((classNameInt+30)%100)/100.0, ((((classNameInt)%1000000)//1000)/100.0)-1.0]).astype(float) #, (classNameInt//1000000)/1000.0]).astype(float)
#            label = className
            label =  className#np.array([((classNameInt+30)%100)/100.0, (classNameInt//1000)/100.0]).astype(float)
#            label = (label*2.0) - 1.0
            if className not in settings.labelnr:
                settings.labelnr[className] = len(settings.labelnr)
            label = [0]*(settings.num_classes)
            label[settings.labelnr[className]] = 1
            labelsb[settings.labelnr[className]] =  className #(((classNameInt//1000000)/1000.0)*2) - 1.0
            labels[i*10000000+j] = label
            images[i*10000000+j] = filename
            j += 1
    #    while i != imagesBlocker:
    #        time.sleep(0.001)
#    print("+", j)
    imagesCombiner.update(images)
    imagesBlocker += 1
#    print("class end: \t", className, "\t",len(images))



def load_photos(directory, names = False):
    global imagesBlocker
    global imagesCombiner
    
#    imagesCombiner.clear()
    threads = []
    images = dict()
    i = 0
    for className in listdir(directory):
        if className[0] != '.':
#            print("class: ", className, i, names)
            if ".png" in className:
                imagesBlocker = 0
                threads.append(threading.Thread(target=loadThisPhoto, args=(directory, className, i)))
            else:
#                threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
                if names == True:
                    loadPhotosNamesInCategory(directory, className, i)
                else:
                    threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
            i += 1
    print(i)
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


def breakTraining():
    while settings.stopTraining != True:  # making a loop
        time.sleep(3)
        key = getkey()
        try:  # used try so that if user pressed other than the given key error will not be shown
            if key == 'q' or key == 'p':  # if key 'q' is pressed
                if key == 'q':
                    settings.stopTraining = True
                    print('\tStopping!\t')
                if key == 'p':
                    settings.shouldShowPlots = True
                    settings.showPlots()
                    print('\tPlots comming!\t')
            else:
                if key != None:
                    print('press q to stop or p to show plots')
                pass
        except:
            print('press q to stop or p to show plots')


keyboardStop = threading.Thread(target=breakTraining)
keyboardStop.start()


#(imagesTrain, imagesTest) = chunks(images, int(len(images)*995/1000))
imagesTest = load_photos(settings.directoryval, names = True).copy()
imagesCombiner.clear()
imagesTrain = load_photos(settings.directorytrain, names = True).copy()
imagesChunks =  createBatch(imagesTrain, 4*(int(settings.batch_size*1.25))+1)


print('Loaded Images: %d / %d' % (int(len(imagesTrain)), int(len(imagesTest))))


# Generators
training_generator = DataGenerator(imagesTrain, labels)
validation_generator = DataGenerator(imagesTest, labels)
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
model_path = os.path.join(settings.save_dir, settings.model_name)
#keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


epoch_start = 0
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
model_path = os.path.join(settings.save_dir, settings.model_name)

settings.model = Sequential()

settings.model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(512, 512, 3)))
#kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))

settings.model.add(Conv2D(32, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(Conv2D(32, (3, 3)))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.001))

settings.model.add(Conv2D(32, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(Conv2D(32, (3, 3)))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.001))

settings.model.add(Conv2D(64, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(Conv2D(64, (3, 3)))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.001))

settings.model.add(Conv2D(128, (5, 5), padding='same'))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(Conv2D(128, (5, 5)))
settings.model.add(LeakyReLU(alpha=0.01))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))
settings.model.add(Dropout(0.00001))

settings.model.add(Conv2D(256, (3, 3), padding='same'))
settings.model.add(LeakyReLU(alpha=0.001))
settings.model.add(Conv2D(256, (3, 3)))
settings.model.add(LeakyReLU(alpha=0.1))
settings.model.add(MaxPooling2D(pool_size=(2, 2)))
settings.model.add(Dropout(0.0001))



settings.model.add(Flatten())

settings.model.add(Dense(settings.num_classes))
#                             , kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
settings.model.add(Activation('linear'))
#    settings.model.add(Dropout(0.0005))
settings.model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
# Let's train the settings.model using RMSprop
#    epoch_start = 0
#settings.model.load_weights(settings.save_dir+"/"+settings.model_name)
#    settings.model.compile(loss='mean_squared_error',
#                  optimizer=opt,
#                  metrics=['mean_squared_error', 'mean_absolute_error'])


settings.model.compile(loss='mean_squared_error',
          optimizer=opt,
          metrics=['categorical_accuracy', 'mean_squared_error', 'mean_absolute_error', 'accuracy'])
filepath=settings.save_dir+"/weights-improvement-"+settings.model_name+"-{epoch:02d}-{categorical_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
webpage = RemoteMonitor(root='http://trees.duszekjk.com', path='/liveupdates/')
callbacks_list = [checkpoint, webpage]
historyAvg = []
if isfile(model_path+".json"):
    try:
        with open(model_path+".json", 'r') as fp:
            historyAvg = json.load(fp)
    except:
        print("error loading history")

History = settings.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1, epochs=settings.epochs, verbose = 2, callbacks=callbacks_list, initial_epoch = epoch_start)

settings.model.save(model_path)
with open(model_path+".json", 'w') as fp:
    json.dump(History.history, fp)
print(History.history.keys())

plt.plot(History.history['categorical_accuracy'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train + test'], loc='upper left')
plt.show()
plt.savefig("plotA.png")
plt.plot(History.history['mean_absolute_error'])
plt.title('model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train + test'], loc='upper left')
plt.show()
plt.savefig("plotB.png")



stopTraining = True


imagesLoaded = loadIMGS(imagesTest)
(x_test, y_test) = np.array(list(imagesLoaded.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesLoaded.keys())])

x_test = x_test.astype('float32')
x_test /= 255.0

scores = settings.model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("additional tests:")
listOfTests = load_photos(settings.directorytest, names = True)
myTest = loadIMGS(listOfTests)
print('Loaded Images Test: %d' % int(len(myTest)))
(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(myTest.keys())])
my_x_test = my_x_test.astype('float32')
my_x_test /= 255.0
#my_y_test = keras.utils.to_categorical(my_y_test)

classes = settings.model.predict(my_x_test, batch_size=16)
j = 0

#print(classes)
if len(classes[0]) == 2:

    arrayX = "["
    arrayY = "["
    for classesProbs in classes:
        trueA = (int(my_y_test[j][0]*100.0)-30.0)/10.0
        trueB = round(my_y_test[j][1], 2)
        predA = (int(classesProbs[0]*100.0)-30.0)/10.0
        predB = round(classesProbs[1], 2)
        print(my_y_test[j], classesProbs)
        print("\ttrue:\t", trueA, trueB, "\tprediction:\t", predA, predB, "\t = ", (abs(trueA - predA)*1000.0//10)/100.0, (abs(trueB - predB)*1000.0//10)/100.0, "file: ", int(100+trueA*10) + 1000 * int(trueB * 100))
        arrayX += str(predA)+", "
        arrayY += str(predB)+", "
        j += 1


    #    print(prediction)
    print(arrayX)
    print(arrayY)

else:
    for classesProbs in classes:
        print(labelsb[np.argmax(my_y_test[j])], labelsb[np.argmax(classesProbs)])
        j += 1
                  

#    print(prediction)
