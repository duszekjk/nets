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
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
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


from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
from sklearn import manifold



from PIL import Image
#import keyboard  # using module keyboard
from getkey import getkey, keys

labelsb = dict()
labels = dict()

imagesCombiner = dict()
imagesBlocker = 0

imagesCombinerLoad = dict()
imagesBlockerLoad = 0

def visualize_embeddings(data_path, model_path, model):
#    model_file = sorted(glob.glob(model_path + '/*.h5'))[-1]

#    print("\n\n" + model_file + "\n\n")

    model.load_weights(model_path)
        
    feature_model = Model(inputs=model.input,
    outputs=model.get_layer('dense_1').output)

    path = data_path + '/test/'
    dirs = listdir(path)

    features = []
    labels = []

#    for i in range(len(dirs)):
#        if(dirs[i][0] != "."):
    img_files = listdir(path)
    for j in range(min(100, len(img_files))):
        if(img_files[j][0] != "." and img_files[j][-4:] != ".csv"):
            img_path = path+img_files[j]
            img = image.load_img(img_path, target_size=(320, 320))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = np.multiply(x, 1./255)
            i = int(img_files[j][:2])
            preds = feature_model.predict(x)[0]
            features.append(preds)
            labels.append(i)

            print(img_files[j], img_files[j][-8:-4])

    tsne = manifold.TSNE(n_components=2, random_state=0)
    projected = tsne.fit_transform(features)

    x = projected[:, 0]
    y = projected[:, 1]

    colors = np.asarray(['black','blue','red','yellow', 'pink', 'green', 'orange', 'purple', 'gray', 'magenta', 'cyan'])
    point_size = 10

    vis_path = data_path + '/vis/'

    plt.clf()
    plt.title('t-SNE')
    plt.scatter(x, y, s=point_size, c=labels,
    cmap=matplotlib.colors.ListedColormap(colors))
    plt.savefig(vis_path + 'tsne.png')
    plt.savefig(vis_path + 'tsne.pdf')




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
        image = load_img(filename, target_size=(320, 320))
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        imagesL[name] = image
    return imagesL

j = 0
def loadThisPhotoNames(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    global j
    images = dict()
            # load an image from file
    photoData = className.split(", ")
    filename = directory + '/' + photoData[0] + ".jpg"
    print(className, filename)
    if photoData[-1] not in settings.labelnr:
        settings.labelnr[photoData[-1]] = len(settings.labelnr)
    label = [0]*(settings.num_classes)
    label[settings.labelnr[photoData[-1]]] = 1
    labelsb[settings.labelnr[photoData[-1]]] =  photoData[-1]
    labels[i*10000000+j] = label
    images[i*10000000+j] = filename
    j += 1
    #    while i != imagesBlocker:
    #        time.sleep(0.001)
#    print("+", j)
    imagesCombiner.update(images)
    imagesBlocker += 1
#    print("class end: \t", className, "\t",len(images))

def loadPhotosNamesInCategory(directory, className, i):
    global imagesBlocker
    global imagesCombiner
    global labels
    global labelsb
    global j
    images = dict()
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



def load_photos(directory, names = False, csv = ""):
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
                print("loading from csv")
                imagesBlocker = 0
#                threads.append(threading.Thread(target=loadThisPhotoNames, args=(directory, className, i)))
                loadThisPhotoNames(directory, className, i)
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
imagesTest = load_photos(settings.directoryval, names = True, csv = settings.directoryval+"/classes.csv").copy()
imagesCombiner.clear()
imagesTrain = load_photos(settings.directorytrain, names = True, csv = settings.directorytrain+"/classes.csv").copy()
imagesCombiner.clear()
#imagesTrainb = load_photos(settings.directorytrainb, names = True).copy()
#imagesTrain.update(imagesTrainb)
imagesChunks =  createBatch(imagesTrain, 4*(int(settings.batch_size*1.25))+1)


print('Loaded Images: %d / %d' % (int(len(imagesTrain)), int(len(imagesTest))))


# Generators
training_generator = DataGenerator(imagesTrain, labels)
validation_generator = DataGenerator(imagesTest, labels)
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
settings.model_name += "CM"
model_path = os.path.join(settings.save_dir, settings.model_name)
#keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


epoch_start = 0
if not os.path.isdir(settings.save_dir):
    os.makedirs(settings.save_dir)
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
settings.model.add(Dropout(0.2, name='dropout_last'))

settings.model.add(Dense(320, name='dense_1b'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_12b'))
settings.model.add(Dropout(0.2))

settings.model.add(Dense(settings.num_classes, name='dense_1'))
settings.model.add(LeakyReLU(alpha=0.01, name='leaky_re_lu_13'))
settings.model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)



#settings.model.load_weights("/Users/jacekkaluzny/Library/Mobile Documents/com~apple~CloudDocs/Studia/📕magisterka AIPD/nets/project/firstNets/saved_models/weights-improvement-5paramitertsR-03-0.9999.hdf5")


settings.model.compile(loss='mean_squared_error',
          optimizer=opt,
          metrics=['categorical_accuracy', 'mean_squared_error', 'mean_absolute_error', 'accuracy'])


filepath=settings.save_dir+"/weights-improvement-"+settings.model_name+"-{epoch:02d}-{categorical_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
webpage = RemoteMonitor(root='http://trees.duszekjk.com', path='/liveupdates/')
callbacks_list = [checkpoint, webpage]

History = settings.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6, epochs=settings.epochs, verbose = 1, callbacks=callbacks_list, initial_epoch = epoch_start)

settings.model.save(model_path)
visualize_embeddings(settings.directory, model_path, settings.model)

stopTraining = True

imagesTestB = load_photos(settings.directorytest, names = True, csv = settings.directorytest+"/classes.csv").copy()

training_generator = DataGenerator(imagesTestB, labels)
#imagesLoaded = loadIMGS(imagesTest)
#(x_test, y_test) = np.array(list(imagesLoaded.values())).reshape(-1,512,512,3), np.array([labels[x] for x in list(imagesLoaded.keys())])
#
#x_test = x_test.astype('float32')
#x_test /= 255.0

#scores = settings.model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
print("additional tests:")
listOfTests = load_photos(settings.directorytest, names = True, csv = settings.directorytest+"/classes.csv").copy()
myTest = loadIMGS(listOfTests)
print('Loaded Images Test: %d' % int(len(myTest)))
(my_x_test, my_y_test) = np.array(list(myTest.values())).reshape(-1,320,320,3), np.array([labels[x] for x in list(myTest.keys())])
my_x_test = my_x_test.astype('float32')
my_x_test /= 255.0
#my_y_test = keras.utils.to_categorical(my_y_test)

classes = settings.model.predict(my_x_test, batch_size=16)
j = 0

print(classes)
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
