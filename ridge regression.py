import numpy as np
import pandas as pd
import random
from sklearn import datasets
from scipy import sparse
from sklearn import preprocessing

filename = "./cpusmall.txt"
X,y = datasets.load_svmlight_file(filename)
X_array = sparse.csr_matrix.todense(X)
X_array = preprocessing.scale(X_array)

n=X_array.shape[0]
m=X_array.shape[1]
#w0=random.randint(0,0,size=(X_array.shape[1],1))
#w0=np.zeros((X_array.shape[1], 1)).astype('int64')
w0 = np.random.uniform(low=-0.5,high=0.5,size=m)
w=w0
step_size=0.0001
#step_size=10**-7,10**-6,10**-5,10**-4,10**-3,10**-2
eps=0.001
lam=1

g0 = 2/n * np.dot(np.transpose(X_array),(np.dot(X_array , w0) - y)) + lam*w0
r0 = np.linalg.norm(g0)

for i in range(1,1000):

    g = 2/n * np.dot(np.transpose(X_array),(np.dot(X_array , w) - y)) + lam*w

    r=np.linalg.norm(g)
    if r<=eps*r0:
        break
    w=w-step_size*g

print(w) 

def mse(X_test,y_test,w):
    n = X_test.shape[0]
    mse=np.linalg.norm(np.dot(X_test, w) - y_test)/n
    return mse

#eps = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2]
MSE_1_1=mse(X_array,y,w)
print(MSE_1_1)
