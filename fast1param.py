#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.applications as kapp


import six
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, serialize
from keras.utils import plot_model
from keras import losses

from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import random
import threading
from itertools import islice

batch_size = 16
removeEveryNBatch = 1
num_classes = 1000.0#len(listdir(directory))
epochs = 2
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'the_tree_model_angles.h5'



labels = []

imagesCombiner = dict()
imagesBlocker = 0

imagesCombinerLoad = dict()
imagesBlockerLoad = 0

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
    images = dict()
    j = 0
#    imagesCombiner = dict()
#    imagesBlocker = 0
    for name in listdir(directory + "/" + className):
        if name[0] != '.':
            # load an image from file
            filename = directory + '/' + className + '/' + name
            classNameInt = int(className)
            label = np.array([(classNameInt+30%100)/100.0, (classNameInt//1000)/100.0]).astype(float)
            labels[i*200+j] = label
            images[i*200+j] = filename
            j += 1
#    while i != imagesBlocker:
#        time.sleep(0.001)
    imagesCombiner.update(images)
    imagesBlocker += 1
    print("class end: \t", className, "\t",len(images))

def loadPhotosInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    images = dict()
    j = 0
    imagesCombiner = dict()
    imagesBlocker = 0
    for name in listdir(directory + "/" + className):
        if name[0] != '.':
            # load an image from file
            filename = directory + '/' + className + '/' + name
            image = load_img(filename, target_size=(512, 512))
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            images[int(className)*100+j] = image
            j += 1
    while i != imagesBlocker:
        time.sleep(0.01)
    imagesCombiner.update(images)
    imagesBlocker += 1
    print("class end: \t", className, "\t",len(images))

def loadThisPhoto(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    images = dict()
    #    print(listdir(directory + "/" + className))
    
    j = 0
    imagesCombiner = dict()
    if className[0] != '.':
        filename = directory + '/' + className
        image = load_img(filename, target_size=(512, 512))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        images[str(i)+str(j)] = image
        j += 1
#    print(i, imagesBlocker)
    while i != imagesBlocker:
        time.sleep(0.01)
    imagesCombiner.update(images)
    imagesBlocker += 1
    print("class end: \t", className, "\t",len(images), imagesBlocker)

def load_photos(directory, names = False):
    global imagesBlocker
    global imagesCombiner
    
    imagesCombiner.clear()
    threads = []
    images = dict()
    i = 0
    for className in listdir(directory):
        if className[0] != '.':            print("class: ", className, i, names)
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
    print(list(imagesCombiner.keys()))
    print("All loaded")
    keys =  list(imagesCombiner.keys())
    random.shuffle(keys)
    for key in keys:
        images[key] = imagesCombiner[key]
    print(list(images.keys()))
    return images

# load images
directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/AP/train'
directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/AP/test'
directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGC/train'
directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGC/test'
#directory = '/Users/jacekkaluzny/Pictures/angles/train'
#directorytest = '/Users/jacekkaluzny/Pictures/angles/test'
#directory = '/Users/jacekkaluzny/Pictures/simplemodel'
#directorytest = '/Users/jacekkaluzny/Pictures/simplemodeltests'
#directory = '/Users/jacekkaluzny/Pictures/fruits/train'
#directorytest = '/Users/jacekkaluzny/Pictures/fruits/test'
#directory = '/Users/jacekkaluzny/Pictures/treenew/tree'
trainDir = directory + "/train"
testDir = directory + "/tests"
images = load_photos(directory, names = True)
#(imagesTrain, imagesTest) = chunks(images, int(len(images)*3/4))
imagesChunks =  createBatch(images, int(batch_size*16/3))
#imagesTrain = load_photos(trainDir)
#imagesTest = load_photos(testDir)
print('Loaded Images: %d' % int(len(images)))

#print(imagesChunks[0])
imagesLoaded = loadIMGS(imagesChunks[0])
#print(list(imagesLoaded.items())[0])
(imagesTrain, imagesTest) = chunks(imagesLoaded, int(len(imagesLoaded)*3/4))
# The data, split between train and test sets:
(x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTrain.keys())]).astype(float)
(x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTest.keys())]).astype(float)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
imagesTrain.clear()
imagesTest.clear()

model = Sequential()



model.add(Conv2D(16, (7, 7), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.005))

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.005))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(1, activation='linear'))
model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss=losses.mean_squared_error,
              optimizer=opt,
              metrics=['mean_squared_error', 'mean_absolute_error'])


for epochNumber in range(1, epochs+1):
    print("\nEpoch ", str(epochNumber)+"/"+str(epochs))
    batchCounter = 0
    for batchData in imagesChunks:
    #    imagesCombinerLoad = dict()
    #    imagesBlockerLoad = 0
#        print(batchData)
        makeSetSmallerInt = random.randint(0, 1000)
        if makeSetSmallerInt > 1000//removeEveryNBatch:
            continue
        imagesLoaded.clear()
        imagesLoaded = loadIMGS(batchData)
#        print(imagesLoaded[0])

        imagesTrain.clear()
        imagesTest.clear()
        (imagesTrain, imagesTest) = chunks(imagesLoaded, int(len(imagesLoaded)*3/4))
        # The data, split between train and test sets:
        (x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTrain.keys())]).astype(float)
        (x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTest.keys())]).astype(float)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0

        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train, epochs=epochNumber, batch_size=batch_size,  verbose=1, validation_split=0.2, initial_epoch = epochNumber-1)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                                         featurewise_center=False,  # set input mean to 0 over the dataset
                                         samplewise_center=False,  # set each sample mean to 0
                                         featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                         samplewise_std_normalization=False,  # divide each input by its std
                                         zca_whitening=False,  # apply ZCA whitening
                                         zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                         # randomly shift images horizontally (fraction of total width)
                                         width_shift_range=0.08,
                                         # randomly shift images vertically (fraction of total height)
                                         height_shift_range=0.08,
                                         shear_range=0.01,  # set range for random shear
                                         zoom_range=0.01,  # set range for random zoom
                                         channel_shift_range=0.5,  # set range for random channel shifts
                                         # set mode for filling points outside the input boundaries
                                         fill_mode='nearest',
                                         cval=0.,  # value used for fill_mode = "constant"
                                         horizontal_flip=True,  # randomly flip images
                                         vertical_flip=True,  # randomly flip images
                                         # set rescaling factor (applied before any other transformation)
                                         rescale=None,
                                         # set function that will be applied on each input
                                         preprocessing_function=None,
                                         # image data format, either "channels_first" or "channels_last"
                                         data_format=None,
                                         # fraction of images reserved for validation (strictly between 0 and 1)
                                         validation_split=0.1)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train,
                                                                              batch_size=batch_size),
                                                                 epochs=epochs,
                                                                 validation_data=(x_test, y_test),
                                                                 workers=2)
        batchCounter += 1

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
plot_model(model, to_file=save_dir+'/model.png')
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("additional tests:")
myTest = load_photos(directorytest)
print('Loaded Images Test: %d' % int(len(myTest)))
#print(myTest[59813])
(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(myTest.keys())]).astype(float)
my_x_test = my_x_test.astype('float32')
my_x_test  /= 255.0
classes = model.predict(my_x_test, batch_size=len(myTest))
j = 0
for classesProbs in classes:
    print("\ttrue:\t", (int(my_y_test[j][0]*100.0)-70)/10.0, int(my_y_test[j][1]*100.0)/100.0, "\tprediction:\t", (int(classesProbs[0]*100.0)-70)/10.0, int(classesProbs[1]*100.0)/100.0, "\t = ", abs(my_y_test[j] - classesProbs[0]), "file: ", int(my_y_test[j][1]*100.0)*1000+int(my_y_test[j][0]*100.0)+30)
    j += 1

    
#    print(prediction)
