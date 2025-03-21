import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(lr, x, y, ep=0.0001, max_iter=1000):
    converged = False
    iter = 0
    m = len(y)

    theta = np.random.random(x.shape[1])
    bias = np.random.random()

    cost_hist = []
    iter_hist = []

    while not converged and iter < max_iter:
        iter += 1
        pred = np.dot(x, theta) + bias
        errors = pred - y

        theta_gradient = (1 / m) *(np.sum (np.dot(x.T, errors)))
        bias_gradient = (1 / m) * np.sum(errors)

        theta = theta - lr * theta_gradient
        bias = bias - lr * bias_gradient

        cost = np.mean(errors**2)
        
        cost_hist.append(cost)
        iter_hist.append(iter)

        if cost <= ep:
            converged = True 

    return theta, bias, cost_hist, iter_hist

if __name__ == '__main__':
    df = pd.read_csv('climate_change.csv')
    training_set = df[df['Year'] <= 2003]
    test_set = df[df['Year'] > 2003]
    train = training_set[['CO2', 'Temp']]
    test = test_set[['CO2', 'Temp']]
    Xtrain = train.drop(['Temp'], axis=1)
    ytrain = train['Temp']
    Xtest = test.drop(['Temp'], axis=1)
    ytest = test['Temp']

    mean_CO2 = Xtrain['CO2'].mean()
    std_CO2 = Xtrain['CO2'].std()
    Xtrain['CO2'] = (Xtrain['CO2'] - mean_CO2) / std_CO2
    Xtest['CO2'] = (Xtest['CO2'] - mean_CO2) / std_CO2

    lr = 0.01
    ep = 0.0001

    theta, bias, cost_history, iteration_history = gradient_descent(lr, Xtrain, ytrain, ep)
    theta = ' + '.join([f'{t} * X{idx+1}' for idx, t in enumerate(theta)])
    print(f'{theta} + {bias}')


    plt.plot(iteration_history, cost_history, color='blue')
    plt.title('Cost Function vs. Iterations (Batch Gradient Descent)')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function (J)')
    plt.show()