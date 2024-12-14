import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def create_cnn_model(input_shape=(224, 224, 3), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the CNN Model
"""def train_model(train_dir, validation_dir, model):

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=30, zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
    val_generator = val_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')

    model.fit(train_generator, epochs=10, validation_data=val_generator)"""