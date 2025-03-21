import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('1D_grades.csv')

g = df.mean()
m = df - g
b = df.std()
df= m / b

bandwidths = [0.1, 0.3, 0.5, 0.7]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histograms with PDF Scatter Plots')

for idx, bandwidth in enumerate(bandwidths):
    mins = df.min()
    maxs = df.max() + bandwidth
    xedge = np.arange(mins['83'], maxs['83'], bandwidth)

    hist = np.zeros(len(xedge) - 1)

    for i in range(len(df)):
        for j in range(len(xedge) - 1):
            if xedge[j] <= df.iloc[i]['83'] < xedge[j + 1]:
                hist[j] += 1

    hist = hist / (len(df) * bandwidth)

    def dens(x, n, xedge, hist, bandwidth):
        for i, d in enumerate(xedge[:-1]):
            if d <= x < xedge[i + 1]:
                num = hist[i]
                return num / (n * bandwidth)
        if x >= xedge[-1]:
            return hist[-1] / (n * bandwidth)
        

    n = len(df)
    x = df['83']
    pdf = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        pdf[i] = dens(x[i], n, xedge, hist, bandwidth)

    axs[idx // 2, idx % 2].scatter(x, pdf/8, color='black')

    axs[idx // 2, idx % 2].bar(xedge[:-1], hist, bandwidth, color='red', edgecolor='black')
    axs[idx // 2, idx % 2].set_xlabel('Normalized Values')
    axs[idx // 2, idx % 2].set_ylabel('Frequency')
    axs[idx // 2, idx % 2].set_title(f'Histogram with Bandwidth = {bandwidth}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
