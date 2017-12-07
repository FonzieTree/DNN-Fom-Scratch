# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
import numpy as np
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
 
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# initialize parameters randomly
h1 = 100 # size of hidden layer
h2 = 100
W1 = 0.01 * np.random.randn(D,h1)
b1 = np.zeros((1,h1))
W2 = 0.01 * np.random.randn(h1,h2)
b2 = np.zeros((1,h2))
W3 = 0.01 * np.random.randn(h2,K)
b3 = np.zeros((1,K))
 
# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
 
# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
   
  # evaluate class scores, [N x K]
  hidden1 = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
  hidden2 = np.maximum(0, np.dot(hidden1, W2) + b2)
  scores = np.dot(hidden2, W3) + b3
   
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
   
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print("iteration %d: loss %f" % (i, loss))
   
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
   
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW3 = np.dot(hidden2.T, dscores)
  db3 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden2 = np.dot(dscores, W3.T)
  # backprop the ReLU non-linearity
  dhidden2[hidden2 <= 0] = 0
  # finally into W,b
  dW2 = np.dot(hidden1.T, dhidden2)
  db2 = np.sum(dhidden2, axis=0, keepdims=True)
  dhidden1 = np.dot(dhidden2,W2.T)
  dhidden1[hidden1 <= 0] = 0
  dW1 = np.dot(X.T, dhidden1)
  db1 = np.sum(dhidden1, axis=0, keepdims=True)
   
  # add regularization gradient contribution
  dW3 += reg * W3
  dW2 += reg * W2
  dW1 += reg * W1
   
  # perform a parameter update
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2
  W3 += -step_size * dW3
  b3 += -step_size * db3
 
# evaluate training set accuracy
hidden1 = np.maximum(0, np.dot(X, W1) + b1)
hidden2 = np.maximum(0, np.dot(hidden1, W2) + b2)
scores = np.dot(hidden2, W3) + b3
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
