from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

n_samples = 500
noise = 0.15
random_state = 42
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.title(f"Moons dataset with {n_samples} samples, noise={noise}, random_state={random_state}")
plt.show()
