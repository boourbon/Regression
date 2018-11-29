import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from scipy import sparse
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def gradient_d(X,y,w0,step_size,eps,lam):
    w = w0
    n = X.shape[0]
    g = 2/n * X.transpose() @ (X @ w - y) + lam*w
    r0 = np.linalg.norm(g)
    for k in range(1,1000):
        g = 2/n * X.transpose() @ (X @ w - y) + lam*w
        if(np.linalg.norm(g) <= eps*r0):
            return w
        w = w - step_size*g
    return w

def mse(X_t,y_t,w):
    n = y_t.shape[0]
    return np.linalg.norm(X_t @ w - y_t)/n
    
filename_train = "./E2006.train.bz2"
filename_test = "./E2006.test.bz2"
X,y = datasets.load_svmlight_file(filename_train)
X_t,y_t = datasets.load_svmlight_file(filename_test)
print(X.shape,y.shape,X_t.shape,y_t.shape)
X_array = X[:,0:150358] #not corresponding

MSE = 0
w0 = np.random.uniform(-0.5,0.5,X_array.shape[1])
step_size=0.001
eps=0.001
lam=1
w = gradient_d(X_array,y,w0,step_size,eps,lam)
MSE = mse(X_t,y_t,w)

print(MSE)
