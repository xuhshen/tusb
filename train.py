# -*- coding:utf-8 -*-
import os 
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


def imagehandle(img):
    '''预处理单张图片
    '''
    return img


def loadsamples(datas="datas"):
    '''导入样本输入输出
    '''
    fnames = [os.path.join(folder[0],f) for folder in os.walk(datas) for f in folder[2]]
    data_y = [int(f.split("_")[0]) for folder in os.walk(datas) for f in folder[2]]
    data_x = np.array([imagehandle(np.load(f)) for f in fnames])
    
    return data_x,data_y

def lstm():
    pass

def cnn(train,test):
    # Generate dummy data
    x_train = train[0]
    y_train = keras.utils.to_categorical(train[1], num_classes=10)
    
    x_test = test[0]
    y_test = keras.utils.to_categorical(test[1], num_classes=10)
    
    print x_train.shape
    
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print score

def train(train,test):
#     x_train = np.random.random((100, 100, 100, 3))
#     y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#     x_test = np.random.random((20, 100, 100, 3))
#     y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    
    cnn(train,test)
    

def predict():
    pass

def show(img):
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":  
    data_x,data_y = loadsamples()
    data_x = data_x.reshape(-1,480,640,1)
    
    train([data_x,data_y],[data_x,data_y])
    
    
    
    
    
    
    
    
    
    