import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
d1=pd.read_csv('banana.csv')
d2=pd.read_csv('blobs.csv')
d3=pd.read_csv('dartboard2.csv')
d4=pd.read_csv('elliptical.csv')
d6=pd.read_csv('twenty.csv')
d1=d1.drop('class',1)
labels=d1['class']
plt.scatter(d1.iloc[:, 0], d1.iloc[:, 1], c=labels, alpha=0.8)
plt.show()
labels=d2['y']
plt.scatter(d2['x1'], d2['x2'], c=labels, alpha=0.8)
plt.show()
labels=d3['class']
plt.scatter(d3.iloc[:, 0], d3.iloc[:, 1], c=labels, alpha=0.8)
plt.show()
plt.scatter(d4.iloc[:, 0], d4.iloc[:, 1], c=labels, alpha=0.8)
plt.show()
label=d6['class']
plt.scatter(d6.iloc[:, 0], d6.iloc[:, 1], c=label, alpha=0.8)
plt.show()
.....kmeans...
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class YourKMeansClass:
    def __init__(self, K):
        self.K = K

    def random_centroids(self, data):
        centroids = []
        for i in range(self.K):
            centroid = data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def new_centroids(self, data, labels):
        centroids = data.groupby(labels).mean().T
        return centroids

    def find_distance(self, data, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
        return distances.idxmin(axis=1)

    def kmeans(self, data, iters):
        centroids = self.random_centroids(data)
        for i in range(iters):
            labels = self.find_distance(data, centroids)
            new_centroids = self.new_centroids(data, labels)
            if np.array_equal(new_centroids, centroids):
                break
            centroids = new_centroids
        return labels, centroids

    def calculate_davies_bouldin_index(self, data, labels, centroids):
        n_cluster = len(np.bincount(labels))
        cluster_k = [data[labels == k] for k in range(n_cluster)]
        centroids = [np.mean(k, axis=0) for k in cluster_k]

        S = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
        Ri = []

        for i in range(n_cluster):
            Rij = []
            for j in range(n_cluster):
                if j != i:
                    r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                    Rij.append(r)
            Ri.append(max(Rij))

        dbi = np.mean(Ri)
        return dbi


    def calculate_davies_bouldin_index(self, data, k_max=20):
        dbi_values = []

        for k_val in range(2, k_max + 1):
            labels, centroids = self.kmeans(data, iters=10)
            dbi = self.compute_davies_bouldin_index(data, labels, centroids)
            dbi_values.append(dbi)

        k_range = np.arange(2, k_max + 1)
        plt.plot(k_range, dbi_values)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Index')
        plt.title('Davies-Bouldin Index')
        plt.show()

    def elbow_plot(self):
        wcss_vals = np.array([])
        for k_val in range(1, self.K):
            labels, centroids = self.kmeans(data, iters=100)
            wcss = 0
            for k in range(self.K):
                wcss += np.sum((data[labels == k_val] - centroids.iloc[:, k]) ** 2)
            wcss_vals = np.append(wcss_vals, wcss)

        K_vals = np.arange(1, self.K)
        plt.plot(K_vals, wcss_vals)
        plt.xlabel('K Values')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.show()
    
    def evaluate_clustering_accuracy(self, true_labels, predicted_labels):
        df = pd.DataFrame({'True_Labels': true_labels, 'Predicted_Labels': predicted_labels})

        percentages = []
        for label in df['True_Labels'].unique():
            same_cluster = df[df['True_Labels'] == label]['Predicted_Labels']
            percentage = (same_cluster == same_cluster.mode().values[0]).mean()
            percentages.append(percentage)

        mean_percentage = np.mean(percentages)

        return mean_percentage
    

# d1 = pd.read_csv('C:\\Users\\dehgh\\OneDrive\\Desktop\\HW-parttern\\HW5\\datasets\\banana.csv')
# data = d1.drop('class', axis=1)
# true_labels=d1['class']
#we use the different data set for each out put we call them sepertaly and show the result 
#this d1 is an instance to show how our code will work
kmeans_instance = YourKMeansClass(K=21) 
kmeans_instance.calculate_davies_bouldin_index(data)
kmeans_instance.elbow_plot()
kmeans_instance.evaluate_clustering_accuracy(t_labels,pred_labels)