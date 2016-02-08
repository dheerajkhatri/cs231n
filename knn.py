import sys
import pickle
import numpy as np


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred


if __name__ == "__main__":
    
    dict  = unpickle(sys.argv[1])    
    Xtr = dict['data']
    Ytr = dict['labels']
    #Xte = dict['batch_label']
    #Yte = dict['filenames']
    
    dict  = unpickle(sys.argv[2])    
    Xte = dict['data']
    Yte = dict['labels']
    
        
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
    
    print Xte.shape[1],Xtr.shape[0]
     
    #nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    #nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    #Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    #print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )                                                                                                         