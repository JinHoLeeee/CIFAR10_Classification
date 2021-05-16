# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 15:20:38 2020
@author: 32183631 Jinho Lee
"""
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
import sys
import urllib.request
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
#image size
img_width, img_height = 32, 32
base_model=VGG16(weights='imagenet',
 include_top=False,
 input_shape=(32,32,3))
nb_epoch=2 #number of epoch
nb_classes=10 #number of labels
#load cifar-10 data
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
last=base_model.get_layer('block5_pool').output
#layers
x=Flatten()(last)
x=Dense(512,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(256,activation='relu')(x)
x=Dropout(0.2)(x)
output=Dense(10,activation='softmax')(x)
model=Model(base_model.input, output)
model.compile(loss='binary_crossentropy',
 optimizer=optimizers.SGD(lr=1e-3,momentum=0.9),
 metrics=['accuracy'])
model.fit(X_train,y_train,
 validation_data=(X_test,y_test),
 nb_epoch=nb_epoch,
 batch_size=200,
 verbose=1)
model.save('CIFAR10-ImageClassification.h5')

# UI file connection
# UI file must be saved in the same directory (with .py file)
# CIFAR-10 labels
class_labels = ['airplain', 'automobile', 'bird', 'cat', 'deer', 'd
                og','frog','horse','ship','truck']
                form_class = uic.loadUiType("test1.ui")[0]
qPixmapVar = QPixmap()  # Need to load image


class WindowClass(QMainWindow, form_class):
 def __init__(self):
  super().__init__()

 self.setupUi(self)
 self.uploadPath.clicked.connect(self.pathFunction)
 self.uploadUrl.clicked.connect(self.urlFunction)

 def pathFunction(self):

 path = self.pathtext.toPlainText()
 qPixmapVar.load(path)
 self.image.setPixmap(qPixmapVar)

 image2 = load_img(path, target_size=(32, 32))
 image2 = img_to_array(image2)


image2 = image2.reshape((1, image2.shape[0], image2.shape[1], ima
                         ge2.shape[2]))
image2 = preprocess_input(image2)
pred = model.predict(image2)
pred = np.argmax(pred, axis=-1)

print(pred)
predResult = class_labels[pred[0]]
print(predResult)
self.result.setText(predResult)


def urlFunction(self):


url = self.urltext.toPlainText()
imageFromWeb = urllib.request.urlopen(url).read()
qPixmapVar.loadFromData(imageFromWeb)
self.image.setPixmap(qPixmapVar)

with urllib.request.urlopen(url) as url:
 with open('C:\\Users\\USER\\Desktop\\2020-
 2\\temp.jpg','wb') as f:
 f.write(url.read())

 image2=load_img('C:\\Users\\USER\\Desktop\\2020-
 2\\temp.jpg',target_size=(32,32))
 image2=img_to_array(image2)

 image2=image2.reshape((1, image2.shape[0], image2.shape[1], ima
 ge2.shape[2]))

 image2=preprocess_input(image2)
 pred=model.predict(image2)
 pred=np.argmax(pred, axis=-1)

predResult=class_labels[pred[0]]
print(predResult)
self.result.setText(predResult)
if __name__ == "__main__":
 # class that can execute the class
 app = QApplication(sys.argv)
 # create instance of WindowClass
 myWindow = WindowClass()
 myWindow.show()
 # execute program
 app.exec_()