# Import necessary libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset and split it into training and testing sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert class labels to binary class matrices
num_classes = 10
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Create a Sequential model
model = Sequential()

# Add a Dense layer with 512 neurons and ReLU activation function as the input layer
model.add(Dense(512, activation='relu', input_shape=(784,)))

# Add a Dropout layer with dropout rate 0.2
model.add(Dropout(0.2))

# Add another Dense layer with 512 neurons and ReLU activation function
model.add(Dense(512, activation='relu'))

# Add another Dropout layer with dropout rate 0.2
model.add(Dropout(0.2))

# Add a Dense layer with num_classes neurons and softmax activation function as the output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data and validate on the test data for 20 epochs with a batch size of 128
model.fit(train_images.reshape(-1, 784), train_labels, validation_data=(test_images.reshape(-1, 784), test_labels),
          epochs=20, batch_size=128)

# Display an image from the test set
plt.imshow(test_images[0], cmap='gray')
plt.show()

# Make a prediction on the first test image and print the predicted class
prediction = model.predict(test_images[0].reshape(1, -1))
print('Model prediction:', np.argmax(prediction))
