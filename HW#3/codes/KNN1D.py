
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
K = [1, 3, 5, 7]
def getDistance(dataPoint, x, k):
    distances = np.array([np.linalg.norm(i-x) for i in dataPoint])

    return np.sort(distances)[k-1]


def KNN(data_1d, k,X_1d):
    N = np.size(data_1d, 0)  
    probs = []  

    for i in x:
        v = np.pi * (getDistance(data_1d,i,k)**2)
        if v == 0:
            probs.append(1)
        else:
            px = k/(N * v)
            probs.append(px)  

    return probs

df = pd.read_csv('1D_grades.csv')
df= df.values
df = data[:,:]
m = df - np.mean(df, axis=0)
b = np.std(df, axis=0)
df = m / b

x = np.linspace(np.min(df[:, 0]), np.max(df[:, 0]), 22).reshape(-1, 1)
probability = []
for k in K:
    probs = KNN(df, k,x)
    plt.plot(x,probs  )
    plt.xlabel('data')
    plt.ylabel('probablity')
    plt.title(f'KNN 1D with k {k} ')
    plt.legend()
    plt.show()  
    