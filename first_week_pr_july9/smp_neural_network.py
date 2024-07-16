import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Define constants
base_dir = '/home/sahitya-jadala/Downloads/1st_week_project/'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')
img_width, img_height = 150, 150  # Image dimensions (adjust as needed)
batch_size = 32
epochs = 10

# Load and preprocess images for training set
train_images = []
train_labels = []
for img_file in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_file)
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    train_images.append(img_array)
    train_labels.append(0)  # Dummy label for training images

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Load and preprocess images for validation set
valid_images = []
valid_labels = []
for img_file in os.listdir(valid_dir):
    img_path = os.path.join(valid_dir, img_file)
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    valid_images.append(img_array)
    valid_labels.append(0)  # Dummy label for validation images

valid_images = np.array(valid_images)
valid_labels = np.array(valid_labels)

# Load and preprocess images for test set
test_images = []
test_labels = []
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    test_images.append(img_array)
    test_labels.append(0)  # Dummy label for test images

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Build the model (ANN)
model = Sequential([
    Flatten(input_shape=(img_width, img_height, 3)),  # Flatten the input (since it's not CNN)
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(valid_images, valid_labels)
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Print individual accuracies
print(f'Training accuracy: {history.history["accuracy"][-1]}')
print(f'Validation accuracy: {history.history["val_accuracy"][-1]}')
