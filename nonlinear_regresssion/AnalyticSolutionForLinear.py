import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import os.path

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


parser = ArgumentParser(description="Analytic solution for linear regression")
parser.add_argument("-i", dest="filename", required=True,
                    help="CSV file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("-c1", dest="x", required=True, help="Name of the column X")
parser.add_argument("-c2", dest="y", required=True, help="Name of the column Y")

args = parser.parse_args()

df = pd.read_csv(args.filename, encoding="utf8")

num_cols = len(list(df))
rng = range(1, num_cols + 1)
new_cols = ['C' + str(i) for i in rng]
df.columns = new_cols[:num_cols]

#X = df.b
#Y = df.c
X_label = args.x   # 'Selling_Price'
Y_label = args.y  # 'Present_Price'
X = df[X_label]
Y = df[Y_label]

# Normal equations for the gradient of the linear function ax + b; obtained on paper to solve the problem by methods of linear algebra
# beta - array of params [A, B]
# X - n input values
# returns matrix A and B for the vector equation Ax=B, i.e. x=inverse_A dot B
def grad(X, Y):
    xx = X.dot(X)
    xs = np.sum(X)
    xy = X.dot(Y)
    ys = np.sum(Y)
    A = np.array([[xx, xs]
                ,[xs, X.size]])
    B = np.array([[xy]
                ,[ys]])
    return A, B
    
A, B = grad(X, Y)
beta = np.linalg.inv(A).dot(B)        
print ("Analytic beta:")
print (beta)

