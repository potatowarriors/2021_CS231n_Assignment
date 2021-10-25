from builtins import range
import numpy as np
from random import shuffle
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    z = np.zeros((num_train, num_class))
    trans_y = np.zeros((num_train,num_class))
    softmax_output = np.zeros((num_train, num_class))
    for i in range(num_train) :
      z[i] = np.dot(X[i],W)
      z[i] = np.exp(z[i])
      trans_y[i][y[i]] = 1 
      temp_sum = np.sum(z[i])
      softmax_output[i] = z[i] / temp_sum
      loss = loss + ((-1) * np.log(softmax_output[i][y[i]]))
      temp_dW = softmax_output[i] - trans_y[i]
      temp_dW = np.reshape(temp_dW,(1,num_class))
      temp_X = X[i]
      temp_X = np.reshape(temp_X,(X.shape[1],1))
      dW = dW + np.dot(temp_X,temp_dW)
      
 
    loss = loss / num_train
    dW = dW / num_train
    loss = loss + ((reg*0.5) * (np.sum(W * W)))
    dW = dW + (reg * W)
        


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    z = np.zeros((num_train, num_class))
    trans_y = np.zeros((num_train,num_class))
    softmax_output = np.zeros((num_train, num_class))
    temp_softmax_matrix = np.zeros((num_train, num_class))
    trans_y[np.arange(num_train), y] += 1
    z = np.dot(X,W)
    z = np.exp(z)
    temp_sum = np.sum(z, axis = 1, keepdims = True)
    softmax_output = z / temp_sum
    loss = np.sum(softmax_output * trans_y)
    loss = loss / num_train
    loss = loss + ((reg*0.5) * (np.sum(W * W)))
    temp_dW = softmax_output - trans_y
    dW = np.dot(X.T,temp_dW)
    dW = dW / num_train
    dW = dW + (reg * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
