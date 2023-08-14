import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG16
from PIL import Image
from numpy import asarray
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing import image
import keras


def getNumPyArray(path,name):
    image = Image.open(path+name)

    image.resize((img_width, img_height))

    numpydata = asarray(image)
    return numpydata

train_size = 150
test_size = 70
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(224, 224, 3))
model = keras.models.load_model('model.keras')

def predImg(dir, name):
    img_path = dir + name  # Replace with the actual path
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    features = conv_base.predict(img_array)
    predictions = model.predict(features)
    return( predictions[0])

print('Displaying test data')

print('potholes')
for i in range(301, 311):
    print (predImg('test/potholes/',str(i)+'.jpg'))
print('normal')
for i in range(301, 311):
    print (predImg('test/normal/',str(i)+'.jpg'))