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
import matplotlib.pyplot as plt

labels = dict()

imagesCombiner = dict()
imagesBlocker = 0


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def loadPhotosInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
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
            classNameInt = int(className)
            label = np.array([((((classNameInt+30)%100)/100.0)*2.0)-1.0, (((classNameInt//1000)/100.0)*2.0)-1.0]).astype(float)
#            print("label:", className, label*1000.0//10)
            labels[i*100+j] = label
            images[i*100+j] = image
            j += 1
    while i != imagesBlocker:
        time.sleep(0.01)
    imagesCombiner.update(images)
    imagesBlocker += 1
    print("class end: \t", (classNameInt%100), (classNameInt//1000), "\t",len(images))

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
        time.sleep(0.001)
    imagesCombiner.update(images)
    imagesBlocker += 1
    print("class end: \t", className, "\t",len(images), imagesBlocker)

def load_photos(directory):
    global imagesBlocker
    global imagesCombiner
    threads = []
    images = dict()
    i = 0
    for className in listdir(directory):
        if className[0] != '.':
#            labels.append(className)
            print("class: ", className)
            if ".png" in className:
                imagesBlocker = 0
                threads.append(threading.Thread(target=loadThisPhoto, args=(directory, className, i)))
            else:
                threads.append(threading.Thread(target=loadPhotosInCategory, args=(directory, className, i)))
#            loadPhotosInCategory(directory, className, i)
            i += 1

    for thread in threads:
        thread.start()
    
    time.sleep(1.0)
    print("class state: \tname\tnr of images (", i, ")")
    while imagesBlocker != i:
        time.sleep(2.0)
        print(imagesBlocker, i)
#    for thread in threads:
#        i -= 1
#        thread.join()
#    images = imagesCombiner
#    imagesCombiner.clear()
#    print(list(imagesCombiner.keys()))
    print("All loaded")
    keys =  list(imagesCombiner.keys())
    random.shuffle(keys)
    for key in keys:
        images[key] = imagesCombiner[key]
    print(list(images.keys()))
    return images

# load images
directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCleafs/train'
directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/APandGCleafs/test'
#directory = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/determinism/train'
#directorytest = '/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/zdjecia drzew/determinism/test'
#directory = '/Users/jacekkaluzny/Pictures/angles/train'
#directorytest = '/Users/jacekkaluzny/Pictures/angles/test'
#directory = '/Users/jacekkaluzny/Pictures/simplemodel'
#directorytest = '/Users/jacekkaluzny/Pictures/simplemodeltests'
#directory = '/Users/jacekkaluzny/Pictures/fruits/train'
#directorytest = '/Users/jacekkaluzny/Pictures/fruits/test'
#directory = '/Users/jacekkaluzny/Pictures/treenew/tree'

images = load_photos(directory)
(imagesTrain, imagesTest) = chunks(images, int(len(images)*3/4))
#imagesTrain = load_photos(trainDir)
#imagesTest = load_photos(testDir)
print('Loaded Images: %d' % int(len(imagesTrain) + len(imagesTest)))

batch_size = 32
num_classes = 1000.0#len(listdir(directory))
epochs = 60
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '2parleafs.h5'

# The data, split between train and test sets:



(x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTrain.keys())]).astype(float)
(x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesTest.keys())]).astype(float)


print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(y_train)

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

model.add(Conv2D(32, (7, 7), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (7, 7)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.005))

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

model.add(Conv2D(128, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.005))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(2, activation='linear'))
model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.00004, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss=losses.mean_squared_error,
              optimizer=opt,
              metrics=['mean_squared_error', 'mean_absolute_error'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0


if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,  verbose=1, validation_split=0.2)
    print(history.history.keys())
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
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

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
#plot_model(model, to_file=save_dir+'/model.png')
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("additional tests:")
myTest = load_photos(directorytest)
print('Loaded Images Test: %d' % int(len(myTest)))
(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(myTest.keys())]).astype(float)
my_x_test = my_x_test.astype('float32')
my_x_test  /= 255.0
classes = model.predict(my_x_test, batch_size=4)
j = 0
arrayX = "["
arrayY = "["
for classesProbs in classes:
    trueA = (int((my_y_test[j][0]+1.0)*50.0)-30.0)/10.0
    trueB = round((my_y_test[j][1]+1.0)*0.5, 2)
    predA = (int((classesProbs[0]+1.0)*50.0)-30.0)/10.0
    predB = round((classesProbs[1]+1.0)*0.5, 2)
    print(classesProbs)
    print("\ttrue:\t", trueA, trueB, "\tprediction:\t", predA, predB, "\t = ", (abs(trueA - predA)*1000.0//10)/100.0, (abs(trueB - predB)*1000.0//10)/100.0, "file: ", int(100+trueA*10) + 1000 * int(trueB * 100))
    arrayX += str(predA)+", "
    arrayY += str(predB)+", "
    j += 1

    
#    print(prediction)
print(arrayX)
print(arrayY)
