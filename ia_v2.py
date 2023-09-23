
import os
import pandas as pd
import numpy as np

from tensorflow import keras
import tensorflow as tf

from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# Charger les données
train_df = pd.read_csv('data/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('data/sign_mnist_test/sign_mnist_test.csv')

# Diviser les données en entrée (pixels) et sortie (étiquettes)
x_train = train_df.iloc[:, 1:].values.astype('float32')
y_train = train_df.iloc[:, 0].values.astype('int32')
x_test = test_df.iloc[:, 1:].values.astype('float32')
y_test = test_df.iloc[:, 0].values.astype('int32')

# Redimensionner les images et normaliser les pixels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Créer un générateur de données pour la formation et la validation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

# Entraîner le modèle en utilisant le générateur de données
batch_size = 32
epochs = 10
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, 
                    validation_data=(x_test, y_test),
                    callbacks=[ModelCheckpoint("model/model-{epoch:03d}.h5", save_best_only=True, save_freq=10, monitor='loss')])

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy : ' + str(test_acc*100) + "%")