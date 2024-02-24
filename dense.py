# Importing necessary libraries
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file named 'diabetes.csv' and convert it to a NumPy array
dataset = pd.read_csv('diabetes.csv', header=None).values

# Split the dataset into features (X) and target labels (Y)
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)

# Set the random seed for reproducibility
np.random.seed(123)

# Initialize a Sequential model
my_first_nn = Sequential() 

# Add a Dense layer with 20 neurons and ReLU activation function as the input layer
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) 

# Add a Dense layer with 4 neurons and ReLU activation function as a hidden layer
my_first_nn.add(Dense(4, activation='relu')) 

# Add a Dense layer with 1 neuron and sigmoid activation function as the output layer
my_first_nn.add(Dense(1, activation='sigmoid')) 

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model on the training data for 25 epochs
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=25, initial_epoch=0)

# Print the summary of the model architecture
print(my_first_nn.summary())

# Evaluate the trained model on the test data and print the test loss and accuracy
print(my_first_nn.evaluate(X_test, Y_test))
