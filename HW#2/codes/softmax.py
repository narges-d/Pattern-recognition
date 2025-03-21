
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.stats as stats

data = np.loadtxt('data.txt')
num_f = (data.shape[1]) - 1
num_c = len(np.unique(data[:, -1]))

x = data[:, :-1]
y = data[:, -1].astype(int)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
def OneHot(y, c):

    y_encoded = np.zeros((len(y), c))

    y_encoded[np.arange(len(y)), y] = 1
    return y_encoded
def Softmax(z):
  exp = np.exp(z - np.max(z))
  for i in range(len(z)):
    exp[i]/=np.sum(exp[i])
  return exp
def fit(X,y, iter_n, lr):

    (m,n) = X.shape

    w = np.zeros((X.shape[1],1))
    b = 0

    loss_arr = []
    Y=OneHot(y,m)
    

    for i in range(1,iter_n+1):

        z = X@w + b
        H=Softmax(z)
        gw = (1/m)*np.dot(X.T,(H - Y))
        gb = (1/m)*np.sum(H - Y)

        w = w - lr * gw
        b = b - lr * gb
        loss = -np.sum(Y * np.log(H) / m )
        if loss>=1 or loss<=0:
            continue
        loss_arr.append(loss)
        if i%2==0:
          if (np.abs (loss-loss_arr[-2]))<=0.000001:
            break

    return w, b, loss_arr


w, b, loss = fit(xtrain, ytrain, iter_n=6000, lr=0.01)
print(len(loss))
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(loss)), loss)
plt.title("Softmax")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()

def predict(X, w, b):
    
    z = X@w + b
    y_hat = Softmax(z)
    
    return np.argmax(y_hat, axis=1)

predictions = predict(xtrain, w, b)
actual_values = ytrain

accuracy = (np.sum(actual_values==predictions)/len(actual_values))*100
print(f'train accuracy is = {accuracy}' )



test_predictions = predict(xtest, w, b)
test_actual = ytest


test_accuracy =( np.sum(test_actual==test_predictions)/len(test_actual))*100

print(f'test accuracy = {test_accuracy}')