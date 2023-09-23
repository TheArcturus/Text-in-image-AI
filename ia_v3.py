
import os
import string
import pandas as pd
import numpy as np

from tensorflow import keras
import tensorflow as tf

from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

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

batch_size = 32
epochs = 10
model = 0

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

# Vérifier si des modèles pré-entraînés existent

model_files = [f for f in os.listdir('model/') if f.endswith('.h5')]

if model_files:

    # Charger le dernier modèle sauvegardé
    latest_model_file = max(model_files)
    model = load_model('model/' + latest_model_file)
    
    # Extraire le numéro de l'époque à partir du nom de fichier
    last_epoch = int(latest_model_file.split('.')[0].split('-')[-1])
    
    # Vérifier si l'entraînement est déjà terminé

    if last_epoch >= epochs:
        print("Le modèle est déjà entraîné jusqu'à l'époque", last_epoch)

    else:

        # Reprendre l'entraînement à partir de la dernière époque

        print("Reprise de l'entraînement à partir de l'époque", last_epoch+1)
        history = model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size), 
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            initial_epoch=last_epoch,
                            epochs=epochs, 
                            validation_data=(x_test, y_test),
                            callbacks=[ModelCheckpoint("model/model-{epoch:03d}.h5", save_best_only=True, save_freq=10, monitor='loss')])

else:

    # Pas de modèle pré-entraîné, entraîner un nouveau modèle

    print("Pas de modèle pré-entraîné trouvé, entraînement d'un nouveau modèle")
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

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

# Faire des prédictions sur l'ensemble de test

num_output = model.output_shape[1]
classes = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
affichage_classes = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
y_pred = model.predict(x_test)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred.argmax(axis=1), labels = classes)

cm_display=ConfusionMatrixDisplay(cm, display_labels=affichage_classes).plot(cmap=plt.cm.Blues)
print(classification_report(y_test,y_pred.argmax(axis=1)))
plt.show()