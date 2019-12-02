import warnings
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

data_path = "./_Data/iris2.csv"

# removes nans
# info - print cleanup info
def cleanup(data, info=True):
    if data.isnull().values.any():
        nulls = data[data.isnull().values]
        val = round(100*float(nulls.shape[0]) / data.shape[0])
        if info == True:
            print('%d/%d (%.2f%%) of NaN rows in the dataset' % (nulls.shape[0], data.shape[0], val))
            print(nulls)
            print('Removing NaN rows')
        data = data.dropna()
    else:
        if info == True:
            print('No NaNs found')
    return data

def categorical(col):
    return col.astype("category").cat.codes
    
def onehot(y, num_classes=0):
    num_classes = np.max(y) + 1 if num_classes == 0 else num_classes
    return np.eye(num_classes)[y]

# analog of sklearn.model_selection.train_test_split
def split(X, y=None, train_size=0.8):
    #numpy.random.shuffle(x)
    #training, test = x[:80,:], x[80:,:]
    indices = np.random.permutation(X.shape[0])
    threshold = int(train_size * 100)
    train_idx, test_idx = indices[:threshold], indices[threshold:]
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = None, None
    if y is not None:
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] 
    return X_train, X_test, y_train, y_test
    
# analog of sklearn.preprocessing.MinMaxScaler
def rescale(X, range=(0, 1), min=None, scale=None):
    if min is None and scale is None:
        minx = np.amin(X, axis=0)
        maxx = np.amax(X, axis=0)
        scale = (range[1] - range[0]) / (maxx - minx)
        min = -minx * scale
    return X * scale + min + range[0], min, scale

# X is a matrix of Nx4 (without bias)
# W is a flattened 5x8 + 9x3 matrices
# returns flattened 8xN + 3xN matrices, where N is a row number of y
def forward(W, X):
    def sigmoid(Z):
        return 1/(1+np.exp(-Z))
        
    size = X.shape[0]
    W1 = W[:5*8].reshape((5, 8))
    W2 = W[5*8:].reshape((9, 3))

    Z1 = W1.T.dot(np.c_[np.ones(size), X].T) # X with added bias
    A1 = sigmoid(Z1)
    Z2 = W2.T.dot(np.r_[[np.ones(size)], A1]) # A1 with added bias
    A2 = sigmoid(Z2)

    return np.concatenate([A1.ravel(), A2.ravel()])

# cross-entropy cost function
# y is a matrix of Nx3
def cross(A, y):
    size = y.shape[0]
    classes = y.shape[1]

    A2 = A[8*size:].reshape((3, size))

    s1 = (y * np.log(A2).T).dot(np.ones(classes)) # wtf did I use this s1 = y.dot(np.log(A2))
    s2 = ((1 - y)* np.log(1 - A2).T).dot(np.ones(classes)) # s2 = (1 - y).dot(np.log(1 - A2))
    return -np.sum(s1 + s2) / size # regularization has been moved out

def regularization(W, size, lmbd):
    return W.dot(W.T)*lmbd/(2*size)
    
# X is a matrix of Nx4 (without bias)
# W is a flattened 5x8 + 9x3 matrices
# A is a flattened 8xN + 3xN matrices, where N is a row number of y
# returns a flattened 5x8 + 9x3 gradient
def backward(W, A, X, y):
    def dsigmoid(z):
        return z * (1 - z)
    size = y.shape[0]
    A1 = A[:8*size].reshape((8, size))
    A2 = A[8*size:].reshape((3, size))
    W1 = W[:5*8].reshape((5, 8))
    W2 = W[5*8:].reshape((9, 3))

    m0_ = A2 - y.T
    G2 = np.r_[[np.ones(size)], A1].dot(m0_.T) / size
    
    m1_ = dsigmoid(A1)
    m2_ = W2[1:,:].dot(m0_)
    m3_= m1_ * m2_
    G1 = m3_.dot(np.c_[np.ones(size), X]).T / size # almost forgot to add bias to X
    
    # regularization is moved out to gradient descent implementations

    return np.concatenate([G1.ravel(), G2.ravel()])

# some info on gradient checking http://cs231n.github.io/neural-networks-3/
# X is a numpy matrix without bias
# lmbd is a regularization parameter. default zero means calculate without regularization
def check_gradient(X, y, epsilon = 1e-4, lmbd=0):
    # The entire W will have N(0,1) distribution
    # That means a standard deviation of Z is neither closer to 1 nor 0, 
    #     hence the neurons of the hidden layer will not get saturated, hence faster learning.
    # See http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
    W = np.random.rand(5*8 + 9*3) # (4 + bias)*8 + (8 + bias)*3
    G_numeric = np.zeros(5*8 + 9*3)
    size = X.shape[0]

    A = forward(W, X)
    G_analytic = backward(W, A, X, y)\
               + W*lmbd/size # accounting for regularization

    id = np.identity(G_numeric.shape[0])
    for i in range(G_numeric.shape[0]):
        W_minus = W - id[i]*epsilon 
        W_plus = W + id[i]*epsilon
        F_minus = cross(forward(W_minus, X), y)\
                + regularization(W_minus, size, lmbd) # accounting for regularization
        F_plus = cross(forward(W_plus, X), y)\
               + regularization(W_plus, size, lmbd) # accounting for regularization
        G_numeric[i] = (F_plus - F_minus)/(2*epsilon)

    err = np.linalg.norm(G_analytic - G_numeric)/max(np.linalg.norm(G_analytic), np.linalg.norm(G_numeric))
    if err > 1e-7:
        raise ValueError('Numerical and analytic gradients differ by order more than 1e-7')
    return 
    
# Gradient descent
# X - one record of X is a row vector, i.e. X is a matrix of the form record num X 4
# y - one-hot vectors 
# lmbd is a regularization parameter. default zero means calculate without regularization
def gd(Xdf, y, nu=0.1, steps=500, lmbd=0):
    # The entire W will have N(0,1) distribution
    # That means a standard deviation of Z is neither closer to 1 nor 0, 
    #     hence the neurons of the hidden layer will not get saturated, hence faster learning.
    # See http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
    W = np.random.rand(5*8 + 9*3) # (4 + bias)*8 + (8 + bias)*3
    F = np.zeros(steps)
    size = Xdf.shape[0]
    
    for i in range(steps):
        A = forward(W, Xdf.to_numpy())
        F[i] = cross(A, y)\
             + regularization(W, size, lmbd) # accounting for regularization
        G = backward(W, A, Xdf.to_numpy(), y)\
          + W*lmbd/size # accounting for regularization
        W -= nu * G # we don't need reshaping before update since G and W are aligned nicely  

    return W, F

def predict(W, Xdf):
    size = Xdf.shape[0]
    A = forward(W, Xdf.to_numpy())
    A2 = A[8*size:].reshape((3, size))
    return A2

# moving average filter to smooth a function
# width is a number of points in a filter.
#       the more points in the window, the smoother the function gets. 
#       though, be mindful that also the bigger the delay (shift to the left) would be
def moving_average(x, width):
    filter_window = np.ones(width)/width
    return np.convolve(x, filter_window, mode='valid')
    
# Stochastic GD 
# y is a one-hot vector
# nu is a gradient step
# batch_size 
# max_epoch_count is a max number of epochs to train. 
#                 May take less if the accuracy on the held-out data would start increasing less than acc_thrs
# validation_hold is a part of Xdf to withhold for use in validation tests
# lmbd is a regularization parameter. default zero means calculate without regularization
# acc_thrs makes gradient descent to stop if the accuracy on the validation data becomes less than this threshold 
# returns 
#   W - a gradient of two matrices 5*8 and 9*3 in one flattened array
#   acc - list of accuracy calculations after each new epoch
def sgd(Xdf, y, nu=0.1, batch_size=10, max_epoch_count=1, validation_hold=0.3, lmbd=0, avg_window=20, acc_thrs=1e-2):
    size = Xdf.shape[0]
    acc = np.zeros(max_epoch_count)
    W = np.random.rand(5*8 + 9*3) # (4 + bias)*8 + (8 + bias)*3
    for i in range(max_epoch_count):
        indices = np.random.permutation(Xdf.shape[0])
        thrsld = int(size*validation_hold)
        idx_train = indices[:size - thrsld]
        idx_valdn = indices[size - thrsld:]
        for j in range(int(idx_train.shape[0]/batch_size)):
            batch_x = Xdf.iloc[idx_train[j*batch_size:(j+1)*batch_size],:]
            batch_y = y[idx_train[j*batch_size:(j+1)*batch_size],:]
            A = forward(W, batch_x.to_numpy())
            G = backward(W, A, batch_x.to_numpy(), batch_y)\
              + W*lmbd/size # accounting for regularization
            W -= nu * G
            
        y_pred = predict(W, Xdf.iloc[idx_valdn,:])
        res = np.nonzero(np.argmax(y[idx_valdn], axis=1) - np.argmax(y_pred.T, axis=1)) # nonzero elements
        acc[i] = 100 - res[0].shape[0] / y[idx_valdn].shape[0] * 100
        
        # calculate numerical derivative of accuracy
        if i >= 2*avg_window:
            avg_pp = moving_average(acc[i-(2*avg_window):i], avg_window)
            acc_deriv_numeric = (avg_pp[-1] - avg_pp[0])/avg_pp.shape[0]
            if np.abs(acc_deriv_numeric) < acc_thrs:
                acc = acc[:i]
                break
        
    return W, acc
        
# Read data
# Inspect
# Transform and clean the data up: remove nulls, transform categorical
# Train-test split
# Normalize train, save normalizer
# Create a model
# Train the model
# Verify the model with test
# Plot learning curves

def print_results(y, y_pred, title):
    res = np.nonzero(y - np.argmax(y_pred.T, axis=1)) # nonzero elements
    print(title + ': %.2f (%i/%i)' % ( 100 - res[0].shape[0] / y.shape[0] * 100
                                     , y.shape[0] - res[0].shape[0]
                                     , y.shape[0]))


data = pd.read_csv(data_path)

X = cleanup(data, False).copy()
X.Species = categorical(X.Species)
y = X.pop('Species')

X_train, X_test, y_train, y_test = split(X, y) 

X_train_scaled, min, scale = rescale(X_train)

check_gradient(X.iloc[[1,2],:], onehot(y_train, num_classes=3)[[1,2],:])

W, cost = gd(X_train_scaled, onehot(y_train, num_classes=3), nu=0.5, steps=10000)

print_results(y_train, predict(W, X_train_scaled), 'Correct predictions on the train set')

X_test_scaled, _, _ = rescale(X_test, min=min, scale=scale)

print_results(y_test, predict(W, X_test_scaled), 'Correct predictions on the test set')

W, acc = sgd(X_train_scaled, onehot(y_train, num_classes=3), nu=0.5, max_epoch_count = 450, lmbd=0.001)

print_results(y_test, predict(W, X_test_scaled), 'Correct predictions (SGD) on the test set')

plt.subplots_adjust(hspace=0.6)
plt.subplot(2, 1, 1)
plt.plot(cost)
plt.title('Learning (GD) on train data')
plt.ylabel('loss')
plt.xlabel('step')
plt.grid(True)
                                                            
plt.subplot(2, 1, 2)
plt.plot(acc)
plt.plot(moving_average(acc, 25), color='red')
plt.title('Accuracy (SGD) on validation data')
plt.ylabel('Accuracy, %')
plt.xlabel('epoch')
plt.grid(True)

plt.show()
