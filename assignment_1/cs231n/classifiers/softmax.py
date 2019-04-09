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
  for i in range(X.shape[0]):
      # scores: shape N,C
      scores = np.dot(X[i,:], W)
      scores = np.exp ( scores - np.max(scores) )
      #normalize scores
      scores /= np.sum(scores)
      
      loss += -np.log ( scores[y[i]] )
      
      scores[y[i]] -= 1
      dW += np.dot ( np.reshape(X[i,:], (-1, 1 ) ) , scores.reshape(1,-1) )
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # Normalize and add regularization
  loss /= X.shape[0]
  loss += reg * np.sum( W**2 )
  dW /= X.shape[0]

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
  scores = np.dot(X,W)
  scores = np.exp (scores - np.max(scores, axis = 1).reshape(-1,1) )
  # normalize scores
  scores /= np.sum(scores, axis = 1).reshape(-1,1)
  
  #extract true label scores for softmax loss
  true_label_scores = scores[np.arange(X.shape[0]), y ]
  loss = np.sum ( -np.log ( true_label_scores ) )
  loss /= X.shape[0]
  loss += np.sum (W**2)
  
  #compute gradient 
  # fix coefficient of scores
  scores[np.arange(X.shape[0]), y] -= 1
  # column - row view of matrix multiplication
  dW = np.dot( X.transpose(), scores )
  dW /= X.shape[0]
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

