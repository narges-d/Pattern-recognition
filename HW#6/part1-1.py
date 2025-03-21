import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
from sklearn.linear_model import LogisticRegression

iris = data.load_iris()
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

selected_classes = [0, 1]
mask = np.isin(iris.target, selected_classes)
x = iris.data[mask][:, [2, 3]]  
y = iris.target[mask]

model = LogisticRegression()
model.fit(x, y)

plt.figure(figsize=(6, 4))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='b', label='Setosa')
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='r', label='Versicolor')

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()
