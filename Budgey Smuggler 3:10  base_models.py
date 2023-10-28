# Mathmatic Libraries
import numpy as np
import pandas as pd

import datetime
import os

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import classification_report, confusion_matrixch

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Layer, Flatten
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


import tensorflow as tf
tf .config.list_physical_devices("GPU")
from tensorflow import keras
from keras.models import Model 
from keras.layers import Dense, GlobalAveragePooling2D, Dropout # BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from keras.models import load_model

#PyTorch??

# Importing deep learning architectures from Keras' applications module.
from keras.applications import VGG16, ResNet50, MobileNet, EfficientNetB0 
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.applications.efficientnet import decode_predictions as efficientnet_decode_predictions




# Load dataset labels
bird_df = pd.read_csv('/Users/uncleosk/Desktop/Budgey_Smuggler/birds_datasets/birds.csv')
# Filter rows where 'data set' column has value 'train'
train_df = bird_df['labels'][bird_df['data set'] == 'train']
# Filter rows where 'data set' column has value 'validation'
valid_df = bird_df['labels'][bird_df['data set'] == 'valid']
# Filter rows where 'data set' column has value 'test'
test_df = bird_df['labels'][bird_df['data set'] == 'test']

train_directory = '/Users/uncleosk/Desktop/Budgey_Smuggler/birds_datasets/train'
test_directory = '/Users/uncleosk/Desktop/Budgey_Smuggler/birds_datasets/test'
valid_directory = '/Users/uncleosk/Desktop/Budgey_Smuggler/birds_datasets/valid'

# Data preprocessing

# Initialize the ImageDataGenerator for augmenting training images.
train_data_gen = ImageDataGenerator(
    rescale = 1./255,              # Normalize pixel values to [0, 1]
    rotation_range = 40,           # Randomly rotate images in the range (0 to 40 degrees)
    width_shift_range = 0.1,      # Randomly shift images horizontally by 10% of total width
    height_shift_range = 0.1,     # Randomly shift images vertically by 10% of total height
    shear_range = 0.3,            # Apply random shear transformations
    zoom_range = 0.1,             # Randomly zoom images by 10%
    horizontal_flip = True,         # Randomly flip images horizontally
    fill_mode = 'constant'        # Fill pixels out of boundaries with a constant value
)

 # Create a data generator for the training set.
train_generator = train_data_gen.flow_from_directory(
    directory=train_directory,       # Main directory containing subdirectories
    target_size=(224, 224),          # Resize images to 224x224
    batch_size=32,                   # Adjust batch size as needed
    class_mode='categorical',        # Multi-class classification mode
    shuffle=True                     # Shuffle the dataset for training
)

# No preporcessing/ augmentation - unaltered images for validation
valid_data_gen = ImageDataGenerator(rescale=1./255)

val_generator = valid_data_gen.flow_from_directory(
    directory=valid_directory,  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# No preporcessing/ augmentation - unaltered images for testing
test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
 directory=test_directory,  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Default fully connected top layer for reference
mobileNet_model = MobileNet(weights='imagenet')

img_path = '/Users/uncleosk/Desktop/Budgey_Smuggler/user_test_images/Masked_Booby.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = mobileNet_model.predict(x)

# Decode the prediction to get human-readable labels
predictions = decode_predictions(preds, top=3)[0]
extracted_data = [(tup[1], tup[2]) for tup in predictions]
for row in extracted_data:
    print('\nImageNet Predicted: ' + row[0] +
           ' - Probability: ' + str(row[1])
        )


# VGG16 Prediction
vgg16_model = VGG16(weights='imagenet')
x_vgg16 = vgg16_preprocess_input(x.copy())
preds_vgg16 = vgg16_model.predict(x_vgg16)
predictions_vgg16 = decode_predictions(preds_vgg16, top=3)[0]
print("\nVGG16 Predictions:")
for (i, (imagenetID, label, prob)) in enumerate(predictions_vgg16):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# ResNet50 Prediction
resnet50_model = ResNet50(weights='imagenet')
x_resnet50 = resnet50_preprocess_input(x.copy())
preds_resnet50 = resnet50_model.predict(x_resnet50)
predictions_resnet50 = decode_predictions(preds_resnet50, top=3)[0]
print("\nResNet50 Predictions:")
for (i, (imagenetID, label, prob)) in enumerate(predictions_resnet50):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# EfficientNetB0 Prediction
efficientnet_model = EfficientNetB0(weights='imagenet')
x_efficientnet = efficientnet_preprocess_input(x.copy())
preds_efficientnet = efficientnet_model.predict(x_efficientnet)
predictions_efficientnet = efficientnet_decode_predictions(preds_efficientnet, top=3)[0]
print("\nEfficientNetB0 Predictions:")
for (i, (imagenetID, label, prob)) in enumerate(predictions_efficientnet):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

keras.backend.clear_session()
