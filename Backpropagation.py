# This is a toy example of conducting backpropagation
# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
from __future__ import division, print_function, absolute_import
import numpy as np
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
# data generation from cs231n.github.io/neural-networks-case-study
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# make one-hot target vector for computing loss function in this example
Y = np.zeros((N*K,K))
for i in range(N):
    Y[i,y[i]] = 1
# some parameters
alpha = 0.01
h = 64
batch_size = 50
lr = 0.5
epoch = 100
num_samples = X.shape[0]
num_dim = X.shape[1]
m = 0.98
batch_size = 1000
num_samples = X.shape[0]
# weight initialization suggestions from cs231n.github.io/neural-networks-case-study
w1 = alpha*(np.random.randn(D*h)*np.sqrt(2.0/(D*h))).reshape(D,h)
w2 = alpha*(np.random.randn(h*K)*np.sqrt(2.0/(h*K))).reshape(h,K)
for i in range(epoch):
    # hidden layer, ReLU activation
    index = np.random.random_integers(num_samples,size=(batch_size)) - 1
    x1 = X[index,:]
    y1 = Y[index,:]
    # feedforwad
    z1 = np.dot(x1,w1)
    h1 = np.maximum(0,z1)
    z2 = np.dot(h1,w2)
    # class score
    yhat = np.exp(z2)
    # compute and normalize class probabilities
    row_max = (np.sum(yhat,axis=1).reshape(batch_size,1))
    yhat = yhat/row_max
    # compute the loss
    correct_logprobs = -np.log(yhat)
    data_loss = np.sum(correct_logprobs*y1)/batch_size
    # backpropagation
    dyhat = (yhat - y1)/batch_size
    dz2 = dyhat
    dh1 = np.dot(dz2,w2.T)
    dw2 = np.dot(h1.T,dz2)
    dh1[np.where(h1<=0)]=0
    dz1 = dh1
    dw1 = np.dot(x1.T,dz1)
    w1 = m*w1 - lr*dw1
    w2 = m*w2 - lr*dw2
    pred=np.argmax(yhat,axis=1)
    act=np.argmax(y1,axis=1)
    acc = sum(act==pred)/batch_size
    print('data_loss: ',data_loss,'    acc: ',acc)
############
print('Done')
