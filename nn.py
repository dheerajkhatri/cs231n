import sys
import pickle
import numpy as np
#import cs231n.data_utils import load_CIFAR10


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
if __name__ == "__main__":
    
    #ie: python knn.py cifar-10-batches-py\data_batch_1 cifar-10-batches-py\test_batch 
    
    print 'Loading Train data'
    #load train data
    dict  = unpickle(sys.argv[1])            
    Xtr = dict['data']   #numpy nd array
    Ytr = dict['labels'] #list of labels
    #Btr = dict['batch_label'] #String
    #Ftr = dict['filenames'] #Filenames of this batch
    
    print 'Training data shape ', Xtr.shape
    print 'Training label shape ', Xtr.shape
    
    print 'Loading Test data'
    #load test data
    dict  = unpickle(sys.argv[2])    
    Xte = dict['data']
    Yte = dict['labels']
    
    print 'Testing data shape ', Xtr.shape
    print 'Testing label shape ', Xtr.shape
        
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 10000 x 3072(for one batch)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xtr_rows becomes 10000 x 3072
        
     
    nn = NearestNeighbor() # create a Nearest Neighbor classifier class
    print 'Training Modle'
    nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
    print 'Testing Model'
    Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )                                                                                                         