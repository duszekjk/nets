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
from keras.callbacks import ModelCheckpoint
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
            classNameInt = int(className)
#            label = np.array([((classNameInt+30)%100)/100.0, ((((classNameInt)%1000000)//1000)/100.0)-1.0]).astype(float) #, (classNameInt//1000000)/1000.0]).astype(float)
#            label = className
            label = np.array([((classNameInt+30)%100)/100.0, (classNameInt//1000)/100.0]).astype(float)
            label = (label*2.0) - 1.0
            labelsb[i*1000+j] =  label #(((classNameInt//1000000)/1000.0)*2) - 1.0
            labels[i*1000+j] = label
            images[i*1000+j] = filename
            j += 1
    #    while i != imagesBlocker:
    #        time.sleep(0.001)
    imagesCombiner.update(images)
    imagesBlocker += 1
#    print("class end: \t", className, "\t",len(images))



def load_photos(directory, names = False):
    global imagesBlocker
    global imagesCombiner
    
    imagesCombiner.clear()
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
                    print('\tPlots comming!\t')
            else:
                if key != None:
                    print('press q to stop or p to show plots')
                pass
        except:
            print('press q to stop or p to show plots')


keyboardStop = threading.Thread(target=breakTraining)
keyboardStop.start()

images = load_photos(settings.directory, names = True)
(imagesTrain, imagesTest) = chunks(images, int(len(images)*995/1000))
imagesChunks =  createBatch(imagesTrain, 4*(int(settings.batch_size*1.25))+1)


print('Loaded Images: %d' % int(len(images)))


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
if isfile(model_path):
    try:
#        del settings.model
#        newModel = load_model(model_path)
#        settings.model = newModel

        settings.model.load_weights("weights.best.hdf5")
        print("model loaded")
    except:
        print("error loading model")
else:

    settings.model = Sequential()

    settings.model.add(Conv2D(16, (3, 3), padding='same',
                     input_shape=(512, 512, 3), kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
        
#    settings.model.add(AvgPooling2D(pool_size=(2, 2)))
    settings.model.add(LeakyReLU(alpha=0.1))
    settings.model.add(Dropout(0.05))
    settings.model.add(MaxPooling2D(pool_size=(2, 2)))

    settings.model.add(Conv2D(16, (3, 3), padding='same'))
    settings.model.add(LeakyReLU(alpha=0.001))
    settings.model.add(Conv2D(16, (3, 3)))
    settings.model.add(LeakyReLU(alpha=0.010))
    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
    settings.model.add(Dropout(0.01))
    
    settings.model.add(Conv2D(32, (3, 3), padding='same'))
    settings.model.add(LeakyReLU(alpha=0.001))
    settings.model.add(Conv2D(32, (3, 3)))
    settings.model.add(LeakyReLU(alpha=0.010))
    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
    settings.model.add(Dropout(0.01))

    settings.model.add(Conv2D(64, (3, 3), padding='same'))
    settings.model.add(LeakyReLU(alpha=0.1))
    settings.model.add(Conv2D(64, (3, 3)))
    settings.model.add(LeakyReLU(alpha=0.1))
    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
    settings.model.add(Dropout(0.001))
#
#    settings.model.add(Conv2D(64, (5, 5), padding='same'))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(Conv2D(64, (5, 5)))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.0001))

#    settings.model.add(Conv2D(128, (5, 5), padding='same'))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(Conv2D(128, (5, 5)))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.001))

#    settings.model.add(Conv2D(256, (5, 5), padding='same'))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(Conv2D(256, (5, 5)))
#    settings.model.add(LeakyReLU(alpha=0.1))
#    settings.model.add(MaxPooling2D(pool_size=(2, 2)))
#    settings.model.add(Dropout(0.05))

    settings.model.add(Flatten())

    settings.model.add(Dense(2, kernel_initializer=keras.initializers.RandomUniform(minval=-1.5, maxval=1.5, seed=random.randint(0, 1000000))))
    settings.model.add(Activation('softmax'))
    settings.model.add(Dropout(0.0005))
    settings.model.summary()

    # initiate RMSprop optimizer
    #opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Let's train the settings.model using RMSprop
#    epoch_start = 0
#    settings.model.load_weights(settings.save_dir+"/weights-improvement-02-0.21.hdf5")
#    settings.model.compile(loss='mean_squared_error',
#                  optimizer=opt,
#                  metrics=['mean_squared_error', 'mean_absolute_error'])


    settings.model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mean_squared_error', 'mean_absolute_error'])
filepath=settings.save_dir+"/weights-improvement-{epoch:02d}-{mean_squared_error:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
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
                    workers=1, epochs=settings.epochs, callbacks=callbacks_list, initial_epoch = epoch_start)

settings.model.save(model_path)
with open(model_path+".json", 'w') as fp:
    json.dump(History.history, fp)
print(History.history.keys())
#plt.plot(History.history['accuracy'])
#plt.title('accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train'], loc='upper left')
#plt.show()
plt.plot(History.history['mean_squared_error'])
plt.title('model loss')
plt.ylabel('mean squared error')
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
(x_test, y_test) = np.array(list(imagesLoaded.values())).reshape(-1,512,512,3), keras.utils.to_categorical(np.array([labels[x] for x in list(imagesLoaded.keys())]), 4)

x_test = x_test.astype('float32')
x_test /= 255.0

scores = settings.model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("additional tests:")
listOfTests = load_photos(settings.directorytest, names = True)
myTest = loadIMGS(listOfTests)
print('Loaded Images Test: %d' % int(len(myTest)))
(my_x_test, my_y_test, my_z_test) = np.array(list(myTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(myTest.keys())]), np.array([labelsb[x] for x in list(myTest.keys())])
my_x_test = my_x_test.astype('float32')
my_x_test /= 255.0


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

    arrayX = "["
    arrayY = "["
    arrayZ = "["
    files = ""
    for classesProbs in classes:

        trueA = (int(((1+my_y_test[j][0])/2.0)*100.0)-30.0)/10.0
        trueB = round(((1+my_y_test[j][1])/2.0), 2)
        predA = (int(((1+classesProbs[0])/2.0)*100.0)-30.0)/10.0
        predB = round(max(((1+classesProbs[1])/2.0), 0.1), 2)
        trueC = round(((1+my_z_test[j])/2.0)*1000)
        predC = trueC
        if len(classesProbs) > 2:
            trueC = round(((1+my_y_test[j][2])/2.0)*1000)
            predC = round(int(((1+classesProbs[2])/2.0)*1000.0), 2)

        print("\tfile: ", str(int(trueC))+str(int(100+trueA*10) + 1000 * int(trueB * 100)), "\ttrue:\t", trueA, trueB, trueC, "\tprediction:\t", predA, predB, predC, "\t = ", (abs(trueA - predA)*1000.0//10)/100.0, (abs(trueB - predB)*1000.0//10)/100.0, abs(trueC-predC), "raw pred:", classesProbs)
        arrayX += str(predA)+", "
        arrayY += str(predB)+", "
        arrayZ += str(predC)+", "
        files += "f\t" + str(int(trueC))+str(int(100+trueA*10) + 1000 * int(trueB * 100)) +"\t"+ str(int(predC))+str(int(100+predA*10) + 1000 * int(predB * 100)) + "\n"
        j += 1

    print(arrayX)
    print(arrayY)
    print(arrayZ)
    print(files)

#    print(prediction)
