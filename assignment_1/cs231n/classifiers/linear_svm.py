import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #count number of rows of W for which w_j . x_i - w_yi . x_i + delta > 0
    # i.e. rows that contribute to loss
    num_rows_contrib_loss = 0    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        num_rows_contrib_loss += 1  
        # contributes to loss = contributes to gradient  
        dW[:,j] += X[i]
        loss += margin
    
    #fix the correct score column in dW
    dW[:,  y[i] ] += -num_rows_contrib_loss * X[i]

  # Divide by N
  dW /= num_train
  dW += 2 * reg * W  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  # get correct scores for each training point
  correct_scores = scores[np.arange(X.shape[0]), y ].reshape(-1, 1)
  #subtract correct score from scores for each training point
  scores = scores - correct_scores + 1
  #subtract the ones we just added to the correct score positions
  scores[np.arange(X.shape[0]), y ] -= 1
  
  #take max (0, ---)
  scores [ scores < 0 ] = 0

  # compute loss
  loss = np.sum(scores)
  loss /= X.shape[0]
  
  # add regularization
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # multipliers for true score column in derivative
  # here scores behaves like a weight matrix: each row gives the weights for
  #X_row in dW
  non_zeros_each_row = np.sum ( scores > 0 , axis = 1 )
  scores[ scores > 0 ] = 1 # making it weights
  scores[ np.arange(X.shape[0]), y ] = -1 * non_zeros_each_row
  
  # the i-th row of scores now contains the recipe for using the i-th row of X
  # to partially construct dW
  # Now remember: [X_col1 X_col2 *[A_row1
  #                                 A_row2]  = X_col1 * A_row1 + X_col2 *A_row2
  # This is exactly what we want except with X transposed. 
  dW = np.dot(X.transpose(), scores)
  dW /= X.shape[0]
  dW += 2 * reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
