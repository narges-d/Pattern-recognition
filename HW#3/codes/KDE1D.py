import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parzen_window(data, x_values, bandwidth):
    
    n = len(data)
    kde_values = np.zeros_like(x_values)

    for i, x in enumerate(x_values):
        # Count the number of data points within the window around x
        count = sum(1 for data_point in data if abs((data_point - x)/bandwidth) <= 1 / 2)

        # Calculate the kernel density at each x value
        kde_values[i] = count / (n * bandwidth)

    return kde_values
def plot_parzen_kde(data, kde_values, x_values):

    fig = plt.figure()

    plt.hist(data, bins=20, density=True, alpha=0.5, color='blue', label='Histogram')  # Plot histogram for reference
    plt.plot(x_values, kde_values, color='red', label='Parzen Window KDE')
    plt.title('Parzen Window Kernel Density Estimation (KDE)')
    plt.xlabel('Data Points')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def Gaussian_kernel( x , x_i , bandwidth):

    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * ((x - x_i) / bandwidth) ** 2)

def kernel_density_estimation(data, x_values, bandwidth):
    
    n = len(data)
    kde_values = np.zeros_like(x_values)

    for i, x in enumerate(x_values):
        # Calculate the kernel density at each x value
        kde_values[i] = (1 / (n * bandwidth)) * sum(Gaussian_kernel(x, x_i, bandwidth) for x_i in data)

    return kde_values

def kde_2d(data, query_points, bandwidth=0.1):
    densities = []

    for query_point in query_points:
        kernel_values = [Gaussian_kernel(query_point, x_i, bandwidth) for x_i in data]
        density_estimate = np.sum(kernel_values) / (len(data) * bandwidth)
        densities.append(density_estimate)

    return np.array(densities)

def plot_kde(data, kde_values, x_values):
    plt.hist(data, bins=20, density=True, alpha=0.5, color='blue', label='Histogram')  # Plot histogram for reference
    plt.plot(x_values, kde_values, color='red', label='KDE')
    plt.title('Kernel Density Estimation (KDE)')
    plt.xlabel('Data Points')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def Main():
    grades_data = pd.read_csv('1D_grades.csv' , header=None , names=['column_1'])
    synthetic_gaussians_data = pd.read_csv('2D_synthetic_gaussians.csv' , header=None  , names=['column_1' , 'column_2'] )
    means = np.mean(grades_data, axis=0)
    stds = np.std(grades_data, axis=0)
    # Normalize the data to zero mean and unit variance
    grades_data = (grades_data - means) / stds
    grades_data = np.array(grades_data)
    synthetic_gaussians_data = (synthetic_gaussians_data-np.mean(synthetic_gaussians_data,axis=0))/np.std(synthetic_gaussians_data , axis=0)
    xn = [x for x in synthetic_gaussians_data.column_1 ]
    yn = [x for x in synthetic_gaussians_data.column_2 ]
    plt.scatter( xn , yn )
    plt.show()
    plt.close()
    
    synthetic_gaussians_data = np.array(synthetic_gaussians_data)
    X_value = np.linspace(min(grades_data) , max(grades_data) ,100)
    bandwidth = 0.5
    kde_value_parzen = parzen_window(grades_data , X_value , bandwidth)
    kde_values = kernel_density_estimation ( grades_data , x_values=X_value , bandwidth=bandwidth)
    plot_parzen_kde (grades_data , kde_values=kde_value_parzen , x_values=X_value)
    plot_kde (grades_data , kde_values=kde_values , x_values=X_value)
    x_value_1 = np.linspace(min(synthetic_gaussians_data[:,0]) , max(synthetic_gaussians_data[:,0]) ,100)
    densities = kde_2d(synthetic_gaussians_data , x_value_1)
    # Plotting the data and KDE estimate
    plt.scatter(synthetic_gaussians_data[:, 0], synthetic_gaussians_data[:, 1], alpha=0.5, label='Data')
    plt.plot(x_value_1, densities, color='blue', label='KDE Estimate')
    plt.title('2D Kernel Density Estimation (KDE) without Library')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    pass


Main()

print('finish!!!')