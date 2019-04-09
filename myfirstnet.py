#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import os
import time

#    instalation keras
#        mac: 
#
#python3 -m venv plaidml-venv
#source plaidml-venv/bin/activate
#
#pip3 install -U plaidml-keras
#
#plaidml-setup
#
#pip3 install plaidml-keras plaidbench
#
#        windows:
#
#            pip install -U plaidml-keras
#            plaidml-setup
#
#os.system("source plaidml-venv/bin/activate")
#
#
#time.sleep(1.0)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.applications as kapp

'''
    #Train a simple deep CNN on the CIFAR10 small images dataset.
    It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
    (it's still underfitting at that point, though).
    '''

#import keras
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

labels = []

imagesCombiner = dict()
imagesBlocker = 0


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def classesProbsToInt(probabilities):
#    i=0
    print(probabilities)
    predictedClassName = np.where(probabilities == np.amax(probabilities))

#    predictedClassNameProb = 0
#    for probability in probabilities:
#        if predictedClassNameProb < probability:
#            predictedClassNameProb = probability
#            predictedClassName = i
#        i += 1

    return predictedClassName[0]

#def classesProbsToIntShape(input_shape):
#    shape = list(input_shape)
##    print(shape)
##    shape /= 180
##    assert len(shape) == 2  # only valid for 2D tensors
#    shape[0] = 1
#    shape[1] = 1
#
#    return input_shape
def my_mean_squared_error(y_true, y_pred):
#    results = np.array([])
#    i = 0
#    for (yt, yp) in zip(y_true, y_pred):
##        i+=1
##        print(i, ",  ")
#        classTrue = classesProbsToInt(yt)
#        classPredicted = classesProbsToInt(yp)
#        print(classPredicted)
#        np.append(results, classPredicted - classTrue)

#    print(y_pred-y_true)
#            return K.mean(K.square(y_pred+(classesProbsToInt(y_pred)-classesProbsToInt(y_true))), axis=-1)
    return K.mean(K.square(y_pred-y_true), axis=-1)

def loadPhotosInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    images = dict()
#    print(listdir(directory + "/" + className))

    j = 0
    imagesCombiner = dict()
    imagesBlocker = 0
    for name in listdir(directory + "/" + className):
#        print("class: ", directory + '/' + className + '/' + name)
        if name[0] != '.': # and (name.split(".")[-1] == "png" or name.split(".")[-1] == "jpg")
            # load an image from file
            filename = directory + '/' + className + '/' + name
#            try:
            image = load_img(filename, target_size=(512, 512))
#            except:
#                continue
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get image id
            #            image_id = name.split('.')[0]
#            print(int(className)*100+j)
            images[int(className)*100+j] = image
            j += 1
    while i != imagesBlocker:
        time.sleep(2.0)
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

def load_photos(directory):
    global imagesBlocker
    global imagesCombiner
    threads = []
    images = dict()
    i = 0
    for className in listdir(directory):
        if className[0] != '.':
            labels.append(className)
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
#directory = '/Users/jacekkaluzny/Pictures/angles/train'
#directorytest = '/Users/jacekkaluzny/Pictures/angles/test'
#directory = '/Users/jacekkaluzny/Pictures/simplemodel'
#directorytest = '/Users/jacekkaluzny/Pictures/simplemodeltests'
#directory = '/Users/jacekkaluzny/Pictures/fruits/train'
#directorytest = '/Users/jacekkaluzny/Pictures/fruits/test'
directory = '/Users/jacekkaluzny/Pictures/treenew/tree'
trainDir = directory + "/train"
testDir = directory + "/tests"
images = load_photos(directory)
#myTest = load_photos(directorytest)
#print('Loaded Images Test: %d' % int(len(myTest)))
(imagesTrain, imagesTest) = chunks(images, int(len(images)*3/4))
#imagesTrain = load_photos(trainDir)
#imagesTest = load_photos(testDir)
print('Loaded Images: %d' % int(len(imagesTrain) + len(imagesTest)))

batch_size = 16
num_classes = 10000#len(listdir(directory))
epochs = 60
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'first_model_angles.h5'

# The data, split between train and test sets:
#tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
#tr_lbl_data = np.array([i[1] for i in training_images])
(x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3).astype(np.uint8), np.array(list(imagesTrain.keys()))
(x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3).astype(np.uint8), np.array(list(imagesTest.keys()))
#(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3).astype(np.uint8), np.array(list(myTest.keys()))
#(x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3).astype(np.uint8), (np.array(list(imagesTrain.keys()))//100).astype(np.uint8)
#(x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3).astype(np.uint8), (np.array(list(imagesTest.keys()))//100).astype(np.uint8)
#(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3).astype(np.uint8), (np.array(list(myTest.keys()))//100).astype(np.uint8)
#(x_train, y_train) = np.array(list(imagesTrain.values())).reshape(-1,512,512,3), (np.array(list(imagesTrain.keys())).astype("int")//100).astype("double")/180.0
#(x_test, y_test) = np.array(list(imagesTest.values())).reshape(-1,512,512,3), (np.array(list(imagesTest.keys())).astype("int")//100).astype("double")/180.0
#(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,512,512,3), (np.array(list(myTest.keys())).astype("int")//100).astype("double")/180.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#y_train = range(1,15)
#y_test = range(1,15)
#my_y_test = range(1,15)

print(y_train)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#my_y_test = keras.utils.to_categorical(my_y_test, num_classes)

print(y_train)

model = Sequential()


model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))
#
#model.add(Conv2D(16, (9, 9), padding='same',
#                 input_shape=x_train.shape[1:]))
#model.add(Activation('relu'))
#model.add(Conv2D(16, (9, 9)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.05))
#
#model.add(Conv2D(32, (3, 3), padding='same',
#                 input_shape=x_train.shape[1:]))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.05))

#model.add(Conv2D(256, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(256, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.05))

model.add(Conv2D(32, (9, 9), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (9, 9)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('relu'))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

# Let's train the model using RMSprop
#model.compile(loss='categorical_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])
#model.compile(loss='mean_squared_error',
#              optimizer=opt,
#              metrics=['mean_squared_error', 'accuracy'])
#model.compile(loss='mean_squared_error',
#              optimizer=opt,
#              metrics=['mean_squared_error', 'accuracy', 'categorical_accuracy'])
model.compile(loss=losses.mean_squared_error,
              optimizer=opt,
              metrics=['poisson', 'mean_absolute_error', 'accuracy'])
#model.compile(loss=my_mean_squared_error,
#              optimizer=opt,
#              metrics=['poisson', 'mean_absolute_error', 'accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#my_x_test = my_x_test.astype('float32')
x_train /= 255
x_test /= 255
#my_x_test  /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
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
plot_model(model, to_file=save_dir+'/model.png')
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("additional tests:")
classes = model.predict(x_test, batch_size=len(myTest))
j = 0
for classesProbs in classes:
    i = 0
    sumProbs = 0
    sum = 0
    predictedClassName = 0
    trueClassName = 0
    predictedClassNameProb = 0
    for probability in classesProbs:
        sumProbs += probability
        sum += probability * i
        if predictedClassNameProb < probability:
            predictedClassNameProb = probability
            predictedClassName = i
        if y_test[j][i] == 1:
            trueClassName = i
        i += 1

    predictedClassNameAvg = sum/sumProbs
    print("true:", trueClassName, "prediction:", predictedClassName)
#    print("true:", my_y_test[j], "prediction:", classesProbs)
#    print(my_y_test[j], classesProbs)
    j += 1

    
#    print(prediction)
