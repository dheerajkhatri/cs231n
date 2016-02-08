import sys
import heapq
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y,k):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y
    self.k = k

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test)

    # loop over all test rows
    for i in xrange(1):      
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) #get distance of current image with all training image
      
      #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1)) #L2 distance
      
      #get list of kmin distances' indices
      k_min_index = heapq.nsmallest(self.k,range(len(distances)),distances.take)  
      
      
      
      #min_index = np.argmin(distances) # get the index with smallest distance
      #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example      

    #return Ypred        