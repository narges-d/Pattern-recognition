import numpy as np
import pandas as pd

def getDistance(dataPoint, x, k):
    distances = np.array([np.linalg.norm(i-x) for i in dataPoint])
    return np.sort(distances)[k-2]


def KNN(data_2d, k,X_2d):
    N = np.size(data_2d, 0)  
    probs = []  

    for x in X_2d:
        v = np.pi * (getDistance(data_2d,x,k)**2)
        if v == 0:
            probs.append(1)
        else:
            px = k/(N * v)
            probs.append(px)  

    return probs

datas = pd.read_csv('2D_synthetic_gaussians.csv')
data= datas.values
fd = data[:,:]
m = fd - np.mean(fd, axis=0)
b = np.std(fd, axis=0)
fd= m / b
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K = [5, 50, 100,200]
x = np.linspace(np.min(fd[:, 0]), np.max(fd[:, 0]), 50).reshape(-1, 1)
y = np.linspace(np.min(fd[:,1]), np.max(fd[:,1]), 50)
y=y.reshape(1,-1)
xx, yy = np.meshgrid(x, y)
X_2d = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
probability = []
for k in K:
    probs = KNN(fd, k,X_2d)
    zz = np.array(probs).reshape(xx.shape)
    probability.append(zz)

for i in range(len(Ks)):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(xx, yy, probability[i], cmap='viridis')
    ax.set_title('KNN - k('+str(K[i])+')')
    fig.tight_layout()






