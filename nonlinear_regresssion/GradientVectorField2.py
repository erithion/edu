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

def is_valid_beta(parser, arg):
    def rg(arg):
        s = arg.split(':')
        if len(s) == 3:
            return np.linspace(float(s[0]), float(s[1]), int(s[2]))
        else:
            return np.linspace(float(s[0]), float(s[0]), 1)
        
    ls = [rg(s) for s in arg.split()]
    if not ls:
        parser.error("An array of betas was either not specified or specified incorrectly!" % arg)
    elif len(ls) - sum(1 if len(s) == 1 else 0 for s in ls) != 2:
        parser.error("More or less than two ranges of betas were specified!" % arg)

    # get 2 ranges
    r = [i for i, y in enumerate(ls) if len(y) > 1]     

    idx_x = r[0]
    idx_y = r[1]
    # generate x-y points from ranges
    xx, yy = np.meshgrid(ls[idx_x], ls[idx_y])
    coords = np.vstack([xx.ravel(), yy.ravel()])
    # fill the gap for the rest of betas
    template_row = np.ones(coords.shape[1])

    # and place every beta array on its place
    ret = np.empty([len(template_row), len(ls)])
    for i in range(0, len(ls)):
        if len(ls[i]) > 1:
            ret[:, i] = coords[0]
            coords = np.delete(coords, 0, 0)
        else:
            ret[:, i] = ls[i][0] * template_row

#    print (ret)
    return ret, xx, yy, idx_x, idx_y


parser = ArgumentParser(description="Builds a gradient vector field for 2 variables")
parser.add_argument("-i", dest="filename", required=True,
                    help="CSV file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("-p", choices=['linear','quadratic','exponential'], required=True, help="type of the used function", type = lambda s : s.lower())
parser.add_argument("-c1", dest="x", required=True, help="Name of the column X")
parser.add_argument("-c2", dest="y", required=True, help="Name of the column Y")
parser.add_argument('-b','--beta', nargs='+', help='<Required> Array of beta integers. Two of them are ranges. Example: 3 1:5:64 2:15:32 16', required=True, type=lambda s: is_valid_beta(parser, s))

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

# Ax + B
# beta - array of params [A, B]
# X - n input values
def linear(beta, X):
    return beta[0]*X + beta[1]
# Jacobian of f
#   column 0 - derivative wrt A
#   column 1 - derivative wrt B
def J_linear(beta, X):
    return np.array([X, np.ones(X.size)])
    

# Ax^2 + Bx + C    
def quadratic(beta, X):
    return beta[0]*np.power(X, 2) + beta[1]*X + beta[2]
    
def J_quadratic(beta, X):
    return np.array([np.power(X, 2), X, np.ones(X.size)])

# Ae^(Bx) + C    
def exponential(beta, X):
    return beta[0]*np.exp(beta[1]*X)+beta[2]
    
def J_exponential(beta, X):
    return np.array([np.exp(beta[1]*X), beta[0]*X*np.exp(beta[1]*X), np.ones(X.size)])
    

def gradient(beta, X, Y, J_f, f):
#    print (beta)
    r_ = Y - f(beta, X)
    j_ = J_f(beta, X)
    return j_.dot(r_)        
    
if args.p == 'linear':
    f = linear
    J = J_linear
elif args.p == 'quadratic':
    f = quadratic
    J = J_quadratic
elif args.p == 'exponential':
    f = exponential
    J = J_exponential
else:
    sys.exit('no linear/quadratic/exponential was specified')

beta, xx, yy, idx_x, idx_y = args.beta[0]

grad = np.array([gradient(b, X, Y, J, f) for b in beta])

#u = 1
#v = -1

u = grad[:, idx_x]
v = grad[:, idx_y]
u_norm = u/np.sqrt(u**2 + v**2)
v_norm = v/np.sqrt(u**2 + v**2)
clr = np.sqrt(u*u + v*v)

qq = plt.quiver(xx, yy, u_norm, v_norm, clr, angles='xy', cmap=plt.cm.jet)
plt.colorbar(qq, cmap=plt.cm.jet)
plt.show()




