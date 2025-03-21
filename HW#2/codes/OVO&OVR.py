
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats

data=np.loadtxt('data.txt')
num_f= (data.shape[1])-1
num_c= len(np.unique(data[:,-1]))

x=data[:,:-1]
y=data[:,-1].astype(int)
x=stats.zscore(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)

def fit_onevsall(X, Y, lr=0.01, itrs=10000):
    W_all = []
    cost_all = []
    X = np.insert(X, 0, 1, axis=1)
    n = len(Y)
    for i in np.unique(Y):
        print('Training:', i, 'vs All')
        W = np.zeros(X.shape[1])
        cost = []
        Y_ovr = np.where(Y == i, 1, 0)
        for k in range(1,itrs+1):
            Z = X.dot(W)
            H = sigmoid(Z)
            W = gradient_desc(X, H, W, Y_ovr, n, lr)
            costs=cal_cost(H, W, Y_ovr)
            cost.append(costs)
            if k%2==0:
                    if (np.abs (costs-cost[-2]))<=0.000001:
                          break
            
        W_all.append((i, W))
        cost_all.append((i, cost))
    return W_all, cost_all

def fit_onevsone(X, Y, lr=0.01, itrs=50000):
    W_all = []
    cost_all = []
    X = np.insert(X, 0, 1, axis=1)
    n = len(Y)
    Y_unq = np.unique(Y)
    m = 0
    for i in range(3):
        for j in range(i+1, 3):
            if i == j:
                continue
            mask = (Y == i) | (Y == j)
            Ynew = Y[mask]
            Xnew = X[mask]
            W = np.zeros(X.shape[1])
            cost = []
            Y_ovo = np.where(Ynew == i, 1, 0)
            for k in range(1,itrs+1):
                Z = Xnew.dot(W)
                H = sigmoid(Z)
                W = gradient_desc(Xnew, H, W, Y_ovo, n, lr)
                costs=cal_cost(H, W, Y_ovo)
                cost.append(costs)
                if k%2==0:
                    if (np.abs (costs-cost[-2]))<=0.000001:
                          break
                
                
            W_all.append((i, j, W))
            cost_all.append((i, j, cost))
            
            m += 1
    return W_all, cost_all

def gradient_desc(X, H, W, Y, n, lr):
    gradient = np.dot(X.T, (H - Y)) / n
    W = W - lr * gradient
    return W

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def cal_cost(H, W, Y):
    n = len(Y)
    cost = (np.sum(-Y.T.dot(np.log(H)) - (1 - Y).T.dot(np.log(1 - H)))) / n
    return cost

def predict_ovr(X, W_all):
    X = np.insert(X, 0, 1, axis=1)
    Y_pred = [max((sigmoid(i.dot(W)), c) for c, W in W_all)[1] for i in X]
    return Y_pred

def predict_ovo(X, W_all):
    X = np.insert(X, 0, 1, axis=1)
    Y_pred = []
    for x in X:
        y_arr = [0 for _ in range(3)]
        for c1, c2, W in W_all:
            res = sigmoid(x.dot(W))
            y_arr[c1] += res
            y_arr[c2] += 1 - res
        Y_pred.append(y_arr.index(max(y_arr)))
    return Y_pred

def cal_score_ovr(X, Y, W_all):
    score = sum(predict_ovr(X, W_all) == Y) / len(Y)
    return score*100

def cal_score_ovo(X, Y, W_all):
    score = sum(predict_ovo(X, W_all) == Y) / len(Y)
    return score*100

def plot_cost_ovr(cost_all):
    for c, cost in cost_all:
        print(len(cost))
        plt.plot(range(len(cost)), cost, label=str(c) + " vs All")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(' Cost Function One VS All')
    plt.legend()
    plt.show()
def plot_ovo(cost_all):
    for c1,c2,cost in cost_all:
        print(len(cost))
        plt.plot(range(len(cost)),cost  )
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(' Cost Function One VS One')
    plt.legend()
    plt.show()  
    
    
W_all_ovr, cost_all_ovr = fit_onevsall(xtrain, ytrain, lr=0.01, itrs=5000)
plot_cost_ovr(cost_all_ovr)

myscore_ovr = cal_score_ovr(xtrain, ytrain, W_all_ovr)
y_pred_ovr = predict_ovr(xtrain, W_all_ovr)
print("Training accuracy (One-vs-All):", myscore_ovr)
myscore2 = cal_score_ovr(xtest,ytest,W_all_ovr)
print("Testing accuracy:",myscore2)

W_all_ovo, cost_all_ovo = fit_onevsone(xtrain, ytrain, lr=0.01, itrs=50000)
plot_ovo(cost_all_ovo)

myscore_ovo = cal_score_ovo(xtrain, ytrain, W_all_ovo)
# y_pred_ovo = predict_ovr(xtrain, W_all_ovo)
print("Training accuracy (One-vs-One):", myscore_ovo)
myscore2 = cal_score_ovo(xtest,ytest,W_all_ovo)
print("Testing accuracy:",myscore2)