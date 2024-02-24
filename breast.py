# Importing necessary libraries
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the Breast Cancer dataset
cancer_data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(cancer_data.data, cancer_data.target,
                                                    test_size=0.25, random_state=87)

# Set the random seed for reproducibility
np.random.seed(155)

# Initialize a Sequential model
my_nn = Sequential()

# Add a Dense layer with 20 neurons and ReLU activation function as the input layer
my_nn.add(Dense(20, input_dim=30, activation='relu'))

# Add a Dense layer with 1 neuron and sigmoid activation function as the output layer
my_nn.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
my_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model on the training data for 100 epochs
my_nn_fitted = my_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# Print the summary of the model architecture
print(my_nn.summary())

# Evaluate the trained model on the test data and print the test loss and accuracy
print(my_nn.evaluate(X_test, Y_test))
