import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
    return np.array(images)

# Define paths
train_path = '/home/sahitya-jadala/Downloads/1st_week_project/train/'
valid_path = '/home/sahitya-jadala/Downloads/1st_week_project/valid/'
test_path = '/home/sahitya-jadala/Downloads/1st_week_project/test/'

# Load data
X_train = load_images_from_folder(train_path)
X_valid = load_images_from_folder(valid_path)
X_test = load_images_from_folder(test_path)


y_train = np.array([0] * len(X_train))  
y_valid = np.array([0] * len(X_valid)) 
y_test = np.array([0] * len(X_test))    

# Define image dimensions
image_width = X_train.shape[1]
image_height = X_train.shape[2]

# If images are grayscale (single channel), adjust accordingly
if len(X_train.shape) == 3:
    X_train = np.expand_dims(X_train, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    num_channels = 1
else:
    num_channels = X_train.shape[3]

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Regularization techniques to try
regularization_techniques = [
    {'name': 'L2 Regularization', 'reg': regularizers.l2(0.001)},
    {'name': 'L1 Regularization', 'reg': regularizers.l1(0.001)},
    {'name': 'Dropout', 'reg': None},
    {'name': 'Batch Normalization', 'reg': None}
]

# Initialize lists to store accuracies
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# Loop through regularization techniques
for reg in regularization_techniques:
    print(f"\nTraining with {reg['name']}:")

    # Example model with regularization technique
    model = Sequential([
        Flatten(input_shape=(image_width, image_height, num_channels)),
        Dense(128, activation='relu', kernel_regularizer=reg['reg']),
    ])

    # Conditionally add Dropout or BatchNormalization layer
    if reg['name'] == 'Dropout':
        model.add(Dropout(0.5))
    elif reg['name'] == 'Batch Normalization':
        model.add(BatchNormalization())

    # Add more layers
    model.add(Dense(64, activation='relu', kernel_regularizer=reg['reg']))

    # Conditionally add Dropout or BatchNormalization layer
    if reg['name'] == 'Dropout':
        model.add(Dropout(0.3))
    elif reg['name'] == 'Batch Normalization':
        model.add(BatchNormalization())

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train-validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Fit the model using Data Augmentation
    batch_size = 32
    steps_per_epoch = len(X_train_split) // batch_size
    history = model.fit(datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=10,
                        validation_data=(X_val, y_val),
                        verbose=0)  # Set verbose to 0 to suppress training output

    # Evaluate on training set
    train_loss, train_acc = model.evaluate(X_train_split, y_train_split, verbose=0)
    train_accuracies.append(train_acc)
    print(f'Training accuracy: {train_acc:.4f}')

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    valid_accuracies.append(val_acc)
    print(f'Validation accuracy: {val_acc:.4f}')

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_accuracies.append(test_acc)
    print(f'Test accuracy: {test_acc:.4f}')

# Print final accuracies for each regularization technique
print("\nFinal Training Accuracies:")
for i, reg in enumerate(regularization_techniques):
    print(f"{reg['name']}: {train_accuracies[i]:.4f}")

print("\nFinal Validation Accuracies:")
for i, reg in enumerate(regularization_techniques):
    print(f"{reg['name']}: {valid_accuracies[i]:.4f}")

print("\nFinal Testing Accuracies:")
for i, reg in enumerate(regularization_techniques):
    print(f"{reg['name']}: {test_accuracies[i]:.4f}")
