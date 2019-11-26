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
    return -np.sum(s1 + s2) / size


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
    
    # TODO: regularization

    return np.concatenate([G1.ravel(), G2.ravel()])

# some info on gradient checking http://cs231n.github.io/neural-networks-3/
# X is a numpy matrix without bias
def check_gradient(X, y, epsilon = 1e-4):
    W = np.random.rand(5*8 + 9*3) # (4 + bias)*8 + (8 + bias)*3
    G_numeric = np.zeros(5*8 + 9*3)

    A = forward(W, X)
    F = cross(A, y)
    G_analytic = backward(W, A, X, y)

    id = np.identity(G_numeric.shape[0])
    for i in range(G_numeric.shape[0]):
        W_minus = W - id[i]*epsilon 
        W_plus = W + id[i]*epsilon
        F_minus = cross(forward(W_minus, X), y)
        F_plus = cross(forward(W_plus, X), y)
        G_numeric[i] = (F_plus - F_minus)/(2*epsilon)

    err = np.linalg.norm(G_analytic - G_numeric)/max(np.linalg.norm(G_analytic), np.linalg.norm(G_numeric))
    if err > 1e-7:
        raise ValueError('Numerical and analytic gradients differ by order more than 1e-7')
    return 
    
# X - one record of X is a row vector, i.e. X is a matrix of the form record num X 4
# y - one-hot vectors 
def train(Xdf, y, nu=0.1, steps=500):
    W = np.random.rand(5*8 + 9*3) # (4 + bias)*8 + (8 + bias)*3
    F = np.zeros(steps)
    
    for i in range(steps):
        A = forward(W, Xdf.to_numpy())
        F[i] = cross(A, y)
        G = backward(W, A, Xdf.to_numpy(), y)
        W -= nu * G # we don't need reshaping before update since G and W are aligned nicely  

    return W, F
    
        
def predict(W, Xdf):
    size = Xdf.shape[0]
    A = forward(W, Xdf.to_numpy())
    A2 = A[8*size:].reshape((3, size))
    return A2

# Read data
# Inspect
# Transform and clean the data up: remove nulls, transform categorical
# Train-test split
# Normalize train, save normalizer
# Create a model
# Train the model
# Verify the model with test
# Plot learning curves

data = pd.read_csv(data_path)

X = cleanup(data, False).copy()
X.Species = categorical(X.Species)
y = X.pop('Species')

X_train, X_test, y_train, y_test = split(X, y) 

X_train_scaled, min, scale = rescale(X_train)

check_gradient(X.iloc[[1,2],:], onehot(y_train, num_classes=3)[[1,2],:])

W, cost = train(X_train_scaled, onehot(y_train, num_classes=3), nu=0.5, steps=10000)

X_test_scaled, _, _ = rescale(X_test, min=min, scale=scale)

y_train_pred = predict(W, X_train_scaled)
res_train = np.nonzero(y_train - np.argmax(y_train_pred.T, axis=1)) # nonzero elements
print('Correct predictions on the train set: %.2f (%i/%i)' % ( 100 - res_train[0].shape[0] / y_train.shape[0] * 100
                                                             , y_train.shape[0] - res_train[0].shape[0]
                                                             , y_train.shape[0]))

y_test_pred = predict(W, X_test_scaled)
res_test = np.nonzero(y_test - np.argmax(y_test_pred.T, axis=1)) # nonzero elements
print('Correct predictions on the test set: %.2f (%i/%i)' % ( 100 - res_test[0].shape[0] / y_test.shape[0] * 100
                                                            , y_test.shape[0] - res_test[0].shape[0]
                                                            , y_test.shape[0]))

plt.plot(cost)
plt.title('Learning curve')
plt.ylabel('cross-entropy cost')
plt.xlabel('steps')
plt.grid(True)
plt.show()
