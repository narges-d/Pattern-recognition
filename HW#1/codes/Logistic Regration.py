import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, w, b, lr, iterations):
    m = X.shape[0]
    costs = []

    for i in range(iterations):
        Z = np.dot(X, w) + b
        A = sigmoid(Z)
        cost = -1/m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        dw = np.sum(np.dot(X.T, (A - y))) / m
        db = np.sum(A - y) / m

        w -= lr * dw
        b -= lr * db
        costs.append(cost)

    return w, b, costs

def predict(X, w, b):
    Z = np.dot(X, w) + b
    A = sigmoid(Z)
    predictions = np.where(A > 0.5, 1, 0)
    return predictions

def logistic_regression(X_train, y_train, X_test, y_test, lr, iterations):
    w = np.zeros((X_train.shape[1], 1))
    b = 0

    w, b, costs = gradient_descent(X_train, y_train, w, b, lr, iterations)

    train_predictions = predict(X_train, w, b)
    test_predictions = predict(X_test, w, b)

    train_accuracy = 100 - np.mean(np.abs(train_predictions - y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_predictions - y_test)) * 100

    print("Train accuracy: {}%".format(train_accuracy))
    print("Test accuracy: {}%".format(test_accuracy))

    a = w[0][0]
    b = b

    equation = "y = ({:} * x1) + {:}".format(a, b)
    print("Logistic regression line:")
    print(equation)

    return w, b, costs

data = pd.read_csv('C:\\Users\\dehgh\\OneDrive\\Desktop\\HW-parttern\\synthetic_software_defect.csv')
X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_test = pd.read_csv('y_test.csv').values

lr = 0.1
iterations = 5000
w, b, costs = logistic_regression(X_train, y_train, X_test, y_test, lr, iterations)

plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function (Training Data)")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c='b', label='Training Set')
plt.scatter(X_test[:, 0], X_test[:, 1], c='r', label='Test Set')

x_decision = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
y_decision = -(w[0] * x_decision + b) / w[1]
plt.plot(x_decision, y_decision, c='g', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training and Test Sets with Decision Boundary')
plt.legend()
plt.show()
