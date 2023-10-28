# Mathmatic Libraries
import numpy as np
import pandas as pd

import datetime
import os

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns

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

# Importing deep learning architectures from Keras' applications module.
from keras.applications import MobileNet 
from keras.applications.mobilenet import preprocess_input, decode_predictions

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

user_test_folder = '/Users/uncleosk/Desktop/Budgey_Smuggler/user_test_images'

def is_name_similar_predictions(img_name, predictions):

    # Removing the file extension from the image name
    name_without_ext = os.path.splitext(img_name)[0]
    
    # Splitting the name into words (assuming names can be like "gray_cuckoo" or "gray cuckoo")
    name_words = set(name_without_ext.replace('_', ' ').split())

    # Checking if any word from the name appears in the predictions
    for label, _ in predictions:
        label_words = set(label.replace('_', ' ').split())
        if name_words & label_words:
            return True
    return False

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Initialize MobileNet architechture model
MobileNet_model = MobileNet(weights='imagenet')


# List all files in the directory
image_files = [f for f in os.listdir(user_test_folder) if os.path.isfile(os.path.join(user_test_folder, f))]

correct_predictions = 0
total_predictions = 0

for img_name in image_files:
    img_path = os.path.join(user_test_folder, img_name)
    
    total_predictions += 1

    try:

        x = preprocess_image(img_path)

        preds = MobileNet_model.predict(x)

        # Decode the prediction to get human-readable labels
        predictions = decode_predictions(preds, top=3)[0]
        extracted_data = [(tup[1], tup[2]) for tup in predictions]
        
        print(f"Predictions for image: {img_name}")
        for row in extracted_data:
            print('ImageNet Predicted:', row[0], '- Probability:', str(row[1]))
        
        if is_name_similar_predictions(img_name, extracted_data):
            correct_predictions += 1
            print("The image name is similar to the predictions!")
        else:
            print("The image name is not similar to the predictions.")
            
        print("\n")
        
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

print(f"Total Correct Predictions: {correct_predictions} out of {total_predictions - 1} total.")