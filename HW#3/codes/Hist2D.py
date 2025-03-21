import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

df = pd.read_csv('2D_synthetic_gaussians.csv')

g = df.mean()
m = df - g
b = df.std()
df= m / b
df.columns = ['a', 'b']

mins_a, maxs_a = df['a'].min(), df['a'].max() + 0.1
mins_b, maxs_b = df['b'].min(), df['b'].max() + 0.1

bandwidths = [0.1, 0.3, 0.5, 0.7]

def calculate_histogram(x, y, xedges, yedges):
    hist = np.zeros((len(xedges)-1, len(yedges)-1))
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            condition = (xedges[i] <= x) & (x < xedges[i+1]) & (yedges[j] <= y) & (y < yedges[j+1])
            hist[i, j] = np.sum(condition)
    return hist

def density(x1, x2, hists, xedges, yedges, n, bandwidth):
    xedges, yedges = xedges[1:], yedges[1:]
    for i, x in enumerate(xedges):
        if x1 < x:
            for j, y in enumerate(yedges):
                if x2 < y:
                    num_samples = hists[i, j]
                    bin_area = bandwidth * bandwidth
                    return num_samples / (n * bin_area)

fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})
fig.suptitle('3D Bar Plots with PDF Values')

for idx, bandwidth in enumerate(bandwidths):
    xedge, yedge = np.arange(mins_a, maxs_a, bandwidth), np.arange(mins_b, maxs_b, bandwidth)
    hist = calculate_histogram(df['a'], df['b'], xedge, yedge)

    xpos, ypos = np.meshgrid(xedge[:-1] + bandwidth / 2, yedge[:-1] + bandwidth / 2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    dx = dy = bandwidth
    dz = np.zeros_like(hist.ravel())

    for i in range(len(xpos)):
        dz[i] = density(xpos[i], ypos[i], hist, xedge, yedge, df.shape[0], bandwidth)

    ax = axs[idx // 2, idx % 2]
    ax.bar3d(xpos, ypos, np.zeros_like(dz), dx, dy, dz, shade=True, color='darkcyan')
    ax.set_xlabel('Normalized Values of a')
    ax.set_ylabel('Normalized Values of b')
    ax.set_zlabel('PDF Value')
    ax.set_title(f'3D Bar Plot with Bandwidth = {bandwidth}')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

