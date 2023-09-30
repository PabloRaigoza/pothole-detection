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


img_width = 224
img_height = 224

def getNumPyArray(path,name):
    image = Image.open(path+name)

    image.resize((img_width, img_height))

    numpydata = asarray(image)
    return numpydata

train_size = 32*20
test_size = 70


conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_width, 3))

x_train = []
y_train = []


base_dir = r"test"


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='binary')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

print('Loading training data...')
train_features, train_labels = extract_features(base_dir, train_size)
print('Training data complete')



model = Sequential()
model.add(Flatten(input_shape=(7,7,512)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_features, train_labels, epochs=20, batch_size=32)
model.save('model.keras')
