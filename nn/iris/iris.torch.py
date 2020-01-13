import warnings
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from score import f1_macro, f1_micro, confusion_matrix, matthews_correlation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


#device = torch.device('cpu')
device = torch.device('cuda') # Uncomment this to run on GPU

# Read data
# Inspect
# Transform and clean the data up: remove nulls, transform categorical
# Train-test split
# Normalize train, save normilizer
# Create a model
# Train the model
# Verify the model with test

# TODO make classifier_torch.py with learning curves and optimal solution search


def inspect(x, y, labels):
    plt.scatter(x, y, c=labels, cmap='viridis')
    plt.show()


# READ DATA  
data_path = "./_Data/iris2.csv"
data = pd.read_csv(data_path)

# INSPECT-CLEANUP-TRANSFORM

if data.isnull().values.any():
    nulls = data[data.isnull().values]
    val = round(100*float(nulls.shape[0]) / data.shape[0])
    print('%d/%d (%.2f%%) of NaN rows in the dataset' % (nulls.shape[0], data.shape[0], val))
    print(nulls)
    print('Removing NaN rows')
    data = data.dropna()
else:
    print('No NaNs found')

# to categorical 
data.Species = data.Species.astype("category").cat.codes
    
print()
print('Inspection of the first five rows')
print(data.head())

# SPLIT
x = data
y = data.pop('Species')

#pdb.set_trace()
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.8, random_state = 5434, shuffle = True) 

# NORMALIZE

#scaler = MinMaxScaler()
#train_x = scaler.fit_transform(train_x)

def onehot(y, num_classes=0):
    return torch.nn.functional.one_hot(y, y.max() + 1)

def print_results(y, y_pred, title):
#    res = np.nonzero(y - np.argmax(y_pred, axis=1)) # nonzero elements
#    print(title + ': %.2f (%i/%i)' % ( 100 - res[0].shape[0] / y.shape[0] * 100
#                                     , y.shape[0] - res[0].shape[0]
#                                     , y.shape[0]))
    m = confusion_matrix(y, y_pred)
    micro = f1_micro(m)
    macro = f1_macro(m)
    matt = matthews_correlation(m)
    res = np.nonzero(y - np.argmax(y_pred, axis=1)) # nonzero elements
    print(title + ': %.2f%% (%i/%i)   Matthews correlation: %.2f   f1_micro: %.2f   f1_macro: %.2f' \
                                     % ( 100 - res[0].shape[0] / y.shape[0] * 100
                                     , y.shape[0] - res[0].shape[0]
                                     , y.shape[0]
                                     , matt
                                     , micro
                                     , macro))
 
 
D_in, D_out, H = 4, 3, 8

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=True),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out, bias=True),
    torch.nn.Sigmoid()
)
loss_fn = torch.nn.BCELoss() # torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()

train_tx = torch.tensor(train_x.values)
train_ty = torch.tensor(train_y.values)

learning_rate = 0.5
count = 10000

ls = np.zeros(count)
for t in range(count):
    y_pred = model(train_tx.float())
#    pdb.set_trace()
    yhot = onehot(train_ty.long()).type_as(y_pred)
    loss = loss_fn(y_pred, yhot)
    ls[t] = loss.item()

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


#pdb.set_trace()            
model.zero_grad()
y_pred = model(torch.tensor(test_x.values).float())

print(y_pred.detach().numpy())
print_results(test_y, y_pred.detach().numpy(), "Accuracy on the test data")
            
plt.plot(ls)
plt.title('Learning on train data')
plt.ylabel('loss')
plt.xlabel('step')
plt.grid(True)
                                                            
plt.show()
