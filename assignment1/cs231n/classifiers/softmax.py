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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train): # Bucle de 0 a N-1

      scores = X[i].dot(W)

      # regularizo scores para evitar la inestabilidad numerica
      scores -= np.max(scores)

      # calculo L_i para esta iteracion
      softmax = np.exp(scores) / np.sum(np.exp(scores))
      loss -= np.log(softmax[y[i]])

      # Actualizamos el gradiente
      for j in range(num_classes):
        dW[:,j] += X[i] * softmax[j]
      dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
 
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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    # calculo los scores (X*W)
    scores = X.dot(W)

    # regularizo scores para evitar la inestabilidad numerica
    scores -= np.max(scores, axis=1).reshape(scores.shape[0], -1)

    # calculo L
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:,None]
    loss = np.sum(-np.log(softmax[np.arange(num_train), y]))

    # Actualizo el gradiente (esta expresión sale de derivar la función L)
    softmax[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
