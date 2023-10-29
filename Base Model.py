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

# Function to reformat input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# List all files in the directory
image_files = [f for f in os.listdir(user_test_folder) if os.path.isfile(os.path.join(user_test_folder, f))]

keras.backend.clear_session()

# Initialise MobileNet model with custom top layers
# Fine-tuning model, custom top layers
base_model = MobileNet(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
    )

# Get the output tensor of the pre-trained model
x = base_model.output 
# Add a global average pooling layer. This helps reduce the spatial dimensions of the output tensor.
x = GlobalAveragePooling2D()(x)
# Add a fully connected (dense) layer with 1024 units and a ReLU activation function.
# This helps the model learn more complex representations.
x = Dense(1024, activation='relu')(x)
# Add a dropout layer with a dropout rate of 0.5. This helps in preventing overfitting.
x = Dropout(0.5)(x)
# Add the final dense layer with 525 units (presumably for 525 classes) with a softmax activation.
# This gives the class probabilities as the output.    

# Output layer
predictions = Dense(525, activation='softmax')(x)
# Construct the final model using the input from the base_model and the newly defined output layer.
model_tuned = Model(inputs=base_model.input, outputs=predictions)
# Loop through all the layers of the base_model and freeze them.
# This ensures that the weights of these layers remain unchanged during training.
for layer in base_model.layers:
    layer.trainable = False

model_tuned.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Define the TensorBoard callback and specify the log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

checkpoint_filepath = 'saved_model/model_Base'

model_checkpoint = ModelCheckpoint(
   filepath=checkpoint_filepath,
   monitor='val_loss',        # What metric to monitor
   save_best_only=True,       # Only save the model that has the best 'val_loss'
   mode='min',                # 'min' means the routine will save the model when 'val_loss' is minimized
   verbose=1                  # Verbosity mode, 1 means it will print logs
)

model_tuned.fit(
    train_generator, 
    epochs=10, 
    validation_data= val_generator,
    callbacks=[early_stopping, tensorboard, reduce_lr, model_checkpoint]
    )

# Using validation data
val_loss, val_accuracy = model_tuned.evaluate(val_generator)
# test data 
test_loss, test_accuracy = model_tuned.evaluate(test_generator)

print("\n")
print(f"Base Validation Loss: {val_loss}")
print(f"Base Validation Accuracy: {val_accuracy}")
print("\n")
print(f"Base Test Loss: {test_loss}")
print(f"Base Test Accuracy: {test_accuracy}")
print("\n")

# Save the trained model - Might want to include for retraining abd better control
model_tuned.save('saved_model/model_Base')

#load the trained model for predictions
loaded_model = load_model('saved_model/model_Base')

# List all files in the directory
image_files = [f for f in os.listdir(user_test_folder) if os.path.isfile(os.path.join(user_test_folder, f))]

correct_predictions = 0
total_predictions = 0

# Getting predictions for the loaded model
for img_name in image_files:
    img_path = os.path.join(user_test_folder, img_name)
    
    total_predictions += 1

    try:
        x = preprocess_image(img_path)

        # Predict using the fine-tuned model
        preds = loaded_model.predict(x)

        # Get the predicted class index
        predicted_class_index = np.argmax(preds[0])

        # Map the class index to its label
        label_map = {v: k for k, v in train_generator.class_indices.items()}
        predicted_label = label_map[predicted_class_index]
        
        print(f"Predictions for image: {img_name}")
        print(f"Model Predicted:", predicted_label, '- Probability:', np.max(preds[0]))
        
        if is_name_similar_predictions(img_name, [(predicted_label, np.max(preds[0]))]):
            correct_predictions += 1
            print("The image name is similar to the predictions!")
        else:
            print("The image name is not similar to the predictions.")
            
        print("\n")
        
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

print(f"Total Correct Predictions: {correct_predictions} out of {total_predictions - 1} total.")


# regularisation - L2: BatchNormalisation
# alternative models such as ResNet and VGG and EfficicentNet
# Data visualisation
# erroy analysis - types of images that are failing
# ensemble models?
# hyperparameter tuning? - Keras tuner? or optuna
# model depth - add layers@!
# Basian optimization - optuna


