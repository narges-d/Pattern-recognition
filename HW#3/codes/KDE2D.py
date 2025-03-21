
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

H = [0.3, 0.5, 0.7, 0.1]

def parzenWindowd(u):
    return np.where(np.abs(u) <= (1/2) ,1,0)

def gaussian(u):
    return (1 / (np.sqrt(2 * np.pi))) * np.exp(-(u**2) / 2)

def KDE(data_2d, h, X_2d, kernel_type='gaussian'):
    N = np.size(data_2d, 0)
    probs = []
    if kernel_type == 'parzen':
        K = parzenWindowd
    elif kernel_type == 'gaussian':
        K = gaussian

    for x in X_2d:
        t = np.sum([K((x - dataPoint) / h) for dataPoint in data_2d.values])
        px = (1 / (N * h)) * t
        probs.append(px)

    return probs


fd = pd.read_csv('2D_synthetic_gaussians.csv')
m = fd - np.mean(fd, axis=0)
b = np.std(fd, axis=0)
fd = m / b

x = np.linspace(np.amin(fd.iloc[:, 0]), np.amax(fd.iloc[:, 0]), 10)
y = np.linspace(np.amin(fd.iloc[:, 1]), np.amax(fd.iloc[:, 1]), 10)
xx, yy = np.meshgrid(x, y)
X_2d = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)

gaussians = []
for h in H:
    probs = KDE(fd, h, X_2d, 'gaussian')
    zz = np.array(probs).reshape(xx.shape)
    gaussians.append(zz)

for i in range(len(gaussians)):
    plt.clf()  
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(xx, yy, gaussians[i], cmap='viridis')
    ax.set_title('Gaussian Kernel - h(' + str(H[i]) + ')')
    fig.tight_layout()
    plt.show()  

for i in range(len(gaussians)):
  
    fig, ax = plt.subplots(1)
    ax.contour(xx, yy, gaussians[i])
    ax.set_title('Gaussian Kernel- h(' + str(H[i]) + ') ')
    fig.tight_layout()
    
    
parzens = []
for h in H:
    probs = KDE(fd,h,X_2d, 'parzen')
    zz = np.array(probs).reshape(xx.shape)
    parzens.append(zz)

print("End Parzen KDE")
for i in range(len(H)):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(xx, yy, parzens[i], cmap='viridis')
    ax.set_title('Parzen Kernel - h('+str(H[i])+')')
    fig.tight_layout()
    plt.show
    
    
for i in range(len(H)):
    fig, ax = plt.subplots(1)
    ax.contour(xx, yy, parzens[i])
    ax.set_title('Parzen Kernel - h('+str(H[i])+')')
    fig.tight_layout()       