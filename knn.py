import sys
import heapq
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y, k):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y
    self.k = int(k)

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test)

    # loop over all test rows
    for i in xrange(num_test):    
      print i,' ..'  
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) #get distance of current image with all training image
      
      #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1)) #L2 distance
      
            
      #get list of kmin distances' indices
      k_min_index = heapq.nsmallest(self.k,range(len(distances)),distances.take)              
      
      #get labels for k min distances
      k_pred_label = []
      for i in xrange(self.k):              
        k_pred_label.append (self.ytr[k_min_index[i]])
      
      #now check if any label is in majority (sort the list by count)      
      sorted_list = sorted(k_pred_label, key=k_pred_label.count, reverse=True)
            
      Ypred[i] = sorted_list[0]      

    return Ypred        