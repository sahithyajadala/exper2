import os
import cv2
import numpy as np

# Load training image data
train_directory = '/home/sahitya-jadala/Downloads/1st_week_project/train'
X_train = []
y_train = []
for filename in os.listdir(train_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as per your images
        image_path = os.path.join(train_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        # Resize image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        X_train.append(image.flatten())  # Flatten image array
        label = 1 if "class1" in filename else -1  # Adjust label based on filename pattern
        y_train.append(label)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Load validation image data
valid_directory = '/home/sahitya-jadala/Downloads/1st_week_project/valid'
X_valid = []
y_valid = []
for filename in os.listdir(valid_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as per your images
        image_path = os.path.join(valid_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        # Resize image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        X_valid.append(image.flatten())  # Flatten image array
        label = 1 if "class1" in filename else -1  # Adjust label based on filename pattern
        y_valid.append(label)

# Convert lists to numpy arrays
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# Load test image data
test_directory = '/home/sahitya-jadala/Downloads/1st_week_project/test'
X_test = []
y_test = []
for filename in os.listdir(test_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as per your images
        image_path = os.path.join(test_directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        # Resize image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        X_test.append(image.flatten())  # Flatten image array
        label = 1 if "class1" in filename else -1  # Adjust label based on filename pattern
        y_test.append(label)

# Convert lists to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Perceptron training on training data only
eta = 0.1  # Learning rate
n_iter = 90  # Number of epochs

# Initialize weights and bias
n_samples, n_features = X_train.shape
weights = np.zeros(n_features)
bias = 0

# Training loop
for _ in range(n_iter):
    for i in range(n_samples):
        linear_output = np.dot(X_train[i], weights) + bias
        y_pred = np.where(linear_output >= 0, 1, -1)
        update = eta * (y_train[i] - y_pred)
        weights += update * X_train[i]
        bias += update

# Calculate training accuracy
predictions_train = np.where(np.dot(X_train, weights) + bias >= 0, 1, -1)
accuracy_train = np.mean(predictions_train == y_train) * 100
print(f'Training Accuracy: {accuracy_train:.2f}%')

# Predict on validation data and calculate accuracy
predictions_valid = np.where(np.dot(X_valid, weights) + bias >= 0, 1, -1)
accuracy_valid = np.mean(predictions_valid == y_valid) * 100
print(f'Validation Accuracy: {accuracy_valid:.2f}%')

# Predict on test data and calculate accuracy
predictions_test = np.where(np.dot(X_test, weights) + bias >= 0, 1, -1)
accuracy_test = np.mean(predictions_test == y_test) * 100
print(f'Test Accuracy: {accuracy_test:.2f}%')

