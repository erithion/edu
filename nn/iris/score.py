import warnings
#import pdb
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#from sklearn.metrics import f1_score
#from sklearn.metrics import matthews_corrcoef


warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# builds a confusion matrix - predicted in rows, real in columns
# y - real values n-vector, y_pred - predicted values n-vector or [n,m]-one-hot matrix. 
#    y and y_pred must have the same row size
def confusion_matrix(y, y_pred):
    dim = len(y_pred.shape)
    assert dim == 1 or dim == 2, "y_pred must be either a one-dimensional class vector or two-dimensional one-hot vector"
    assert y_pred.shape[0] == y.shape[0], "y_pred and y must have the same number of rows"
    
    if dim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    class_num = np.max(y) + 1
    ret = np.zeros((class_num, class_num))
    for i, v in enumerate(y):
        ret[v, y_pred[i]] += 1
    return ret.T

# calculates f1 score for each class and averages it
def f1_macro(conf_mtx):
    precision =  conf_mtx.diagonal() / conf_mtx.sum(dtype=np.float_, axis=1)
    recall =  conf_mtx.diagonal() / conf_mtx.sum(dtype=np.float_, axis=0)
    f1 = 2*precision*recall/(precision + recall)
    return f1.sum() / f1.shape[0]

# uses overall precision and overall recall amongst all classes to compute a harmonic mean (f1 score)
def f1_micro(conf_mtx):
    precision = conf_mtx.diagonal().sum() / conf_mtx.sum(dtype=np.float, axis=1).sum()
    recall = conf_mtx.diagonal().sum() / conf_mtx.sum(dtype=np.float, axis=0).sum()
    return 2*precision*recall/(precision + recall) # harmonic mean
    
# Use 
#  y_real = ...
#  y_predicted = ...
#  matrix = confusion_matrix(y_real, y_predicted)
#  micro_average = f1_micro(matrix)
#                    # the same in sklearn.metrics: micro_average = f1_score(y_real, y_predicted, average='micro')
#  macro_average = f1_macro(matrix)
                     # the same in sklearn.metrics: macro_average = f1_score(y_real, y_predicted, average='macro')

                     
# from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882#s2
#   or https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
# supports multiclass
#   conf_mtx - confusion matrix
# Use
#   matt = matthews_correlation(conf_mtx)
#       # the same in sklearn.metrics:     matt = matthews_corrcoef(y_real, y_predicted)
def matthews_correlation(conf_mtx):
    c = conf_mtx.diagonal().sum()
    s = conf_mtx.sum()
    p = conf_mtx.sum(axis=1) # vector of the number of prediction of each class 
    t = conf_mtx.sum(axis=0) # vector of the number of real occurence of each class
    
    ss = s*s
    pp = p*p
    tt = t*t
    return (c*s - (p*t).sum()) / np.sqrt((ss - pp.sum())*(ss - tt.sum()))
