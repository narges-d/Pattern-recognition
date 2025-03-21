import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.svm import SVC

df = data.load_iris()
x = df.data[:, 2:4] 
y = df.target

indices = np.where((y == 0) | (y == 1))
x = x[indices]
y = y[indices]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

def plot_svm (clf, X_train, y_train, X_test, y_test):
    """
    Generate a simple plot of SVM including the decision boundary, margin, and its training/test data
    
    Parameters
    ----------
    clf: your classifier handle (after training)
    X: feature matrix shape(m_samples, n_features)
    y: label vector shape(m_samples, )
        for both train and test
    """
    # Create a mesh grid based on the provided axes (100 x 100 resolution)
    x0s = np.linspace(min(X_train[:,0])-0.5,max(X_train[:,0])+0.5, 100)
    x1s = np.linspace(min(X_train[:,1])-0.5,max(X_train[:,1])+0.5, 100)
    x0, x1 = np.meshgrid(x0s,x1s) # create a mesh grid
    X_mesh = np.c_[x0.ravel(), x1.ravel()] # convert all mesh points into 2-D points
    y_pred = clf.predict(X_mesh).reshape(x0.shape) # predict then covert back to the 2-D
    y_decision = clf.decision_function(X_mesh).reshape(x0.shape)

    plt.figsize=(16, 9)
    # plot the training set 
    plt.plot(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], "bo", label="Class 0")
    plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], "go", label="Class 1")

    # plot the test set 
    plt.plot(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], "bx")
    plt.plot(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], "gx")

    # Plot out the support vectors (in red)
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80, c="r", label="Support Vectors")
    # Plot decision boundary and margins
    plt.contourf(x0,x1, y_pred, cmap = plt.cm.brg, alpha = 0.1)
    plt.contourf(x0,x1, y_decision, cmap = plt.cm.brg, alpha = 0.2)
    plt.contour(x0, x1, y_decision, colors='k',
                 levels=[-1, 0, 1], alpha=0.5,
                 linestyles=['--', '-', '--'])
    plt.legend(loc="lower right")
    plt.axis("auto")

    plt.grid(True, which='both')
    plt.xlabel("x1")
    plt.ylabel("x2") 


def tune(model, params, X_train, y_train):
    rnd_search = RandomizedSearchCV(model, param_distributions =params, n_iter = 50, cv = 5, random_state=40)
    rnd_search.fit(X_train, y_train)
    print("best hyper-parameter value: ", rnd_search.best_params_)
    bestmodel = rnd_search.best_estimator_
    return bestmodel

model = SVC(kernel='linear')  
parameters = {"C": uniform(0, 20)}
bestmodel = tune(model, parameters, xtrain, ytrain)
pred = bestmodel.predict(xtest)
recall = recall_score(ytest, pred)  
precision = precision_score(ytest, pred)  
conf_matrix = confusion_matrix(ytest, pred)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix:\n", conf_matrix)
plot_svm(bestmodel, xtrain, ytrain, xtest, ytest)
plt.show()