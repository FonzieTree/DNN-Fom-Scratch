# this example is mainly revised from cs231n.github.io/neural-networks-case-study 
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
 
h = 32 # size of hidden layer
W1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
 
# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
batch_size = 50
epoch = 10000
# gradient descent loop
num_examples = X.shape[0]
 
for i in range(epoch):
  # select random bacth
  index = np.random.random_integers(num_examples,size=(batch_size)) - 1
  x1 = X[index,:]
  y1 = y[index]
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(x1, W1) + b1) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
   
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
   
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(batch_size),y1])
  data_loss = np.sum(corect_logprobs)/batch_size
  reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 100 == 0:
    print("iteration %d: loss %f" % (i, loss))
   
  # compute the gradient on scores
  dscores = probs
  dscores[range(batch_size),y1] -= 1
  dscores /= batch_size
   
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W1,b1
  dW1 = np.dot(x1.T, dhidden)
  db1 = np.sum(dhidden, axis=0, keepdims=True)
   
  # add regularization gradient contribution
  dW2 += reg * W2
  dW1 += reg * W1
   
  # perform a parameter update
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2
 
 
# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
