import numpy as np
from past.builtins import xrange

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
  num_class = W.shape[1]
  for i in xrange(num_train):
    y_i = y[i]
    scores = X[i].dot(W)
    scores -= scores.max()
    exp_scores = np.exp(scores)
    L_i = -scores[y_i] + np.log(exp_scores.sum())
    loss += L_i
    denorm = exp_scores.sum()
    for c in xrange(num_class):
      dW[:, c] += exp_scores[c]*X[i]/denorm
      if c == y_i:
        dW[:, c] -= X[i]
  loss /= num_train
  loss += (0.5*reg*W*W).sum()

  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  # loss
  score = X.dot(W) # NxC
  score -= score.max(axis=1, keepdims=True)
  prob = np.exp(score) #NxC
  loss = (-score[range(num_train), y] + np.log(prob.sum(axis=1)))
  loss = loss.sum()/num_train
  loss += (0.5*reg*W*W).sum()
  
  # gradient
  norm_prob = prob/prob.sum(axis=1, keepdims=True) # NxC
  norm_prob[range(num_train), y] -= 1
  dW = X.T.dot(norm_prob)/num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

