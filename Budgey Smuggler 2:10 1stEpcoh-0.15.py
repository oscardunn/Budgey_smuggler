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
from keras.applications import VGG16, ResNet50, InceptionV3, MobileNet 
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
    

keras.backend.clear_session()

# Fine-tuning model, custom top layers
base_model = MobileNet(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
    )

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(525, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Define the TensorBoard callback and specify the log directory
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001)

#checkpoint_filepath = 'saved_model/my_bird_model'

#model_checkpoint = ModelCheckpoint(
#    filepath=checkpoint_filepath,
#    monitor='val_loss',        # What metric to monitor
#    save_best_only=True,       # Only save the model that has the best 'val_loss'
#    mode='min',                # 'min' means the routine will save the model when 'val_loss' is minimized
#    verbose=1                  # Verbosity mode, 1 means it will print logs
#)

model.fit(
    train_generator, 
    epochs=15, 
    validation_data= val_generator,
    # callbacks=[early_stopping], tensorboard_callback, reduce_lr] # model_checkpoint]  # Multiple callbacks
    )


# Unfreeze the top 20 layers
#for layer in model.layers[-15:]:
 #   layer.trainable = True


#model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(
 #   train_generator, 
  #  epochs=45, 
   # validation_data=val_generator,
    #callbacks=[early_stopping, tensorboard_callback,] #reduce_lr, model_checkpoint]
    #)



# Using validation data
val_loss, val_accuracy = model.evaluate(val_generator)
# test data 
test_loss, test_accuracy = model.evaluate(test_generator)

print("\n")
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
print("\n")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print("\n")



# Save the trained model - Might want to include for retraining abd better control
# model.save('saved_model/my_bird_model')




# load the trained model for predictions
# loaded_model = load_model(checkpoint_filepath)


# function to make predictions on an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def img_file_type(filename):
    # Check if the given filename is an image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    # Check if any valid extension matches the file extension
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

# This provides a mapping from class index to class label
class_labels = list(train_generator.class_indices.keys())

user_test_directory = '/Users/uncleosk/Desktop/Budgey_Smuggler/user_test_images'
image_files = [f for f in os.listdir(user_test_directory) if img_file_type(f)]

for img_file in image_files:
    img_path = os.path.join(user_test_directory, img_file)
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(img_path)
        

        # USE LOADED MODEL WHEN SAVED

        # Make predictions
        predictions = model.predict(processed_img)
        
        # Get the class with the highest confidence
        predicted_class_index = np.argmax(predictions[0])
        
      # Get the class with the highest confidence
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]  # Map index to label
        confidence = predictions[0][predicted_class_index]
        
        print(f"Image: {img_file}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}\n")
    
    except Exception as e:
        print(f"Error processing image {img_file}: {str(e)}")


# regularisation - L2: BatchNormalisation
# alternative models such as ResNet and VGG and EfficicentNet
# Data visualisation
# other eval matrics?
# erroy analysis - types of images that are failing
# ensemble models?
# hyperparameter tuning? - Keras tuner? or optuna
# model depth - add layers@!

# MUST SAVE model to prevent retraining - learn how to save then recall!
# Tensor board can monitor progress - learn

