
import pandas as pd
import numpy as np

from tensorflow import keras
import tensorflow as tf

from keras import layers
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

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

# Créer le modèle
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Créer un générateur de données pour la formation et la validation
train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=3,
    brightness_range=[0.85, 1.15],
    zoom_range=[0.9, 1.1],
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

# Entraîner le modèle en utilisant le générateur de données
train_generator = train_data_gen.flow(x_train, y_train, batch_size=64, subset='training')
validation_generator = train_data_gen.flow(x_train, y_train, batch_size=64, subset='validation')

# Callback pour sauvegarder l'entraînement toutes les 10 epochs
model_checkpoint = ModelCheckpoint("model/model-{epoch:03d}.h5", save_best_only=True, save_freq=10, monitor='loss')

# Entraîner le modèle en utilisant le générateur de données et la sauvegarde de modèle
model.fit(train_generator, validation_data=validation_generator, epochs=100, callbacks=[model_checkpoint])

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(x_test)

# Convertir les prédictions en étiquettes
y_pred_labels = np.argmax(y_pred, axis=1)

# Afficher la matrice de confusion
confusion_matrix = tf.math.confusion_matrix(y_test, y_pred_labels)
print(confusion_matrix)