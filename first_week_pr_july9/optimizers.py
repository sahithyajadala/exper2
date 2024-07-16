import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_path = '/home/sahitya-jadala/Downloads/1st_week_project/train/'
valid_path = '/home/sahitya-jadala/Downloads/1st_week_project/valid/'
test_path = '/home/sahitya-jadala/Downloads/1st_week_project/test/'

# Optimizers to try
optimizers = [
    {'name': 'Adam', 'optimizer': Adam()},
    {'name': 'RMSprop', 'optimizer': RMSprop()},
    {'name': 'SGD', 'optimizer': SGD()}
]

# Initialize lists to store accuracies
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# Loop through optimizers
for opt in optimizers:
    print(f"\nTraining with {opt['name']} optimizer:")

    # Load data for this optimizer
    X_train = []
    y_train = []
    for filename in os.listdir(train_path):
        img = Image.open(os.path.join(train_path, filename))
        if img is not None:
            X_train.append(np.array(img))
            y_train.append(0)  # Replace with actual labels loading

    X_valid = []
    y_valid = []
    for filename in os.listdir(valid_path):
        img = Image.open(os.path.join(valid_path, filename))
        if img is not None:
            X_valid.append(np.array(img))
            y_valid.append(0)  # Replace with actual labels loading

    X_test = []
    y_test = []
    for filename in os.listdir(test_path):
        img = Image.open(os.path.join(test_path, filename))
        if img is not None:
            X_test.append(np.array(img))
            y_test.append(0)  # Replace with actual labels loading

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

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

    # Example model without regularization
    model = Sequential([
        Flatten(input_shape=(image_width, image_height, num_channels)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    # Compile the model with optimizer
    model.compile(optimizer=opt['optimizer'],
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

# Print final accuracies for each optimizer
print("\nFinal Training Accuracies:")
for i in range(len(optimizers)):
    print(f"{optimizers[i]['name']}: {train_accuracies[i]:.4f}")

print("\nFinal Validation Accuracies:")
for i in range(len(optimizers)):
    print(f"{optimizers[i]['name']}: {valid_accuracies[i]:.4f}")

print("\nFinal Testing Accuracies:")
for i in range(len(optimizers)):
    print(f"{optimizers[i]['name']}: {test_accuracies[i]:.4f}")
