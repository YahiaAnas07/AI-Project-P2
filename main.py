import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, num_features):
        self.weights = np.random.uniform(-1, 1, size=num_features)
        self.bias = np.random.uniform(-1, 1)

    def predict(self, features):
        activation = 0
        for i in range(len(features)):
            activation += features[i] * self.weights[i]
        return 1 if activation >= 0 else 0

    def accuracy(self, data):
        correct_predictions = sum(1 for features, label in data if self.predict(features) == label)
        total_samples = len(data)
        return correct_predictions / total_samples

    def train(self, training_data, target_accuracy=0.75, learning_rate=0.1):
        epochs = 0
        while True:
            epochs += 1
            for features, label in training_data:
                prediction = self.predict(features)
                error = label - prediction
                self.bias += learning_rate * error
                self.weights += learning_rate * error * features

            current_accuracy = self.accuracy(training_data)
            print(f"Epoch {epochs}, Accuracy: {current_accuracy}")

            if current_accuracy >= target_accuracy:
                print("Training complete. Reached target accuracy.")
                break


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to binary classification: 1 for 'Iris-versicolor' and 0 for others
y_binary = (y == 1).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Convert training data to the required format for perceptron training
training_data = list(zip(X_train, y_train))

# Create a perceptron with the number of features
num_features = X_train.shape[1]
perceptron = Perceptron(num_features=num_features)

# Train the perceptron
perceptron.train(training_data, target_accuracy=0.75)

# Test the perceptron on the test set
test_data = list(zip(X_test, y_test))
test_accuracy = perceptron.accuracy(test_data)
print(f"Test Accuracy: {test_accuracy}")
