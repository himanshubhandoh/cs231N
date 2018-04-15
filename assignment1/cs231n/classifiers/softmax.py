import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    max_score = np.max(scores)
    reduced_scores = scores - max_score
    sum_of_exp = np.sum(np.exp(reduced_scores))
    correct_class_score = reduced_scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        loss = loss - reduced_scores[j]
        dW[:,j] = dW[:,j] - X[i]
      dW[:,j]+=(np.exp(reduced_scores[j])/sum_of_exp) * X[i]
    loss += np.log(sum_of_exp)
    
  loss /= num_train
  dW /= (1.0*num_train)
  

  loss += reg * np.sum(W * W)  
  dW += 2*reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = X.shape[0]
  C = W.shape[1]
  
  max_per_example = np.max(scores, axis = 1)
  reduced_scores = scores - max_per_example[np.newaxis].T
  all_exponents = np.exp(reduced_scores)
  one_hot_targets = np.eye(C)[y]
  loss-= np.sum(reduced_scores * one_hot_targets) #adds the correct class scores
  all_exponents_row_sum = np.sum(all_exponents, axis = 1)
  loss+=np.sum(np.log(all_exponents_row_sum))
  loss/= num_train
  loss+=reg*np.sum(W*W)

  class_multipliers = all_exponents/all_exponents_row_sum[np.newaxis].T
  class_multipliers-=one_hot_targets
  dW = X.T.dot(class_multipliers)*1.0/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

