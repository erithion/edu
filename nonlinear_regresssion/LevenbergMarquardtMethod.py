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


parser = ArgumentParser(description="Regression with Levenberg-Marquardt method")
parser.add_argument("-i", dest="filename", required=True,
                    help="CSV file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
parser.add_argument("-p", choices=['linear','quadratic','exponential'], required=True, help="type of the used function", type = lambda s : s.lower())
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
    
# f - a linear/polynomial/etc. function with a signature func(beta, X)
# J_f - a Jacobian of the function f above
# X and Y - input variables of the set
# beta - initial parameters to be optimized
def leven_marq(beta, X, Y, J_f, f):
#    beta = np.ones(beta.size)
#    pdb.set_trace()

    def residual(beta, X, Y, J_f, f):
        J = J_f(beta, X)
        dist = Y - f(beta, X)
        return dist.T.dot(dist)
        
    def step(beta, X, y, J_f, f, lmd):
        J = J_f(beta, X)
        dist = Y - f(beta, X)
        JJ = J.T.dot(J)
        pinv = np.linalg.pinv(JJ + lmd * np.identity(X.size))
        return pinv.dot(J.T).T.dot(dist)
        
    lmd = 1
    down = 3
    up = 2
    sum = np.array([residual(beta, X, Y, J_f, f)])
#    pdb.set_trace()
    while True:
        # optima ?
        if (sum.size > 1 and np.abs((sum[-2] - sum[-1])/sum[-2]) < 0.0001):
            break

        shift1 = step(beta, X, Y, J_f, f, lmd)
        shift2 = step(beta, X, Y, J_f, f, lmd / down)
        
        cost1 = residual(beta + shift1, X, Y, J_f, f)
        cost2 = residual(beta + shift2, X, Y, J_f, f)
#        print('Sum[-1=] %f' % sum[-1])
#        print('Cost1 %f' % cost1)
#        print('Cost2 %f' % cost1)
        
        if (sum[-1] < cost1 and sum[-1] < cost2):
            lmd *= up
            continue
        
        if (cost2 < sum[-1]):
            lmd /= down
            sum = np.append(sum, cost2)
            print('Cost %d' % cost2)
            beta += shift2
            continue
            
        if (cost2 > sum[-1] and cost1 < sum[-1]):
            sum = np.append(sum, cost1)
            print('Cost %d' % cost1)
            beta += shift1
            continue
            
        break # cost is seems to be a stable point

    return beta, sum

if args.p == 'linear':
    N = 2
    f = linear
    J = J_linear
elif args.p == 'quadratic':
    N = 3
    f = quadratic
    J = J_quadratic
elif args.p == 'exponential':
    N = 3
    f = exponential
    J = J_exponential
else:
    sys.exit('no linear/quadratic/exponential was specified')

beta = np.ones(N)    
beta, profile = leven_marq(beta, X, Y, J, f)        
print ("Fit parameters beta:")
print (beta)

#fig2 = plt.figure(2)
plt.scatter(X, Y)
x_ = np.sort(X)
y_ = f(beta, x_)
plt.plot(x_, y_, color='red')
plt.title('Fit with Gauss-Newton method')
plt.ylabel(Y_label)
plt.xlabel(X_label)
plt.grid(True)

plt.figure(2)
ax = plt.subplot(111)

plt.scatter(range(0, len(profile)), profile, color='black')
offs = 0.05 * np.amax(profile)
for xy in zip(range(0, len(profile)), profile):                                       # <--
    ax.annotate('%.1f' % xy[1], xytext=(xy[0], xy[1] + offs), xy=xy, textcoords='data',
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle"), bbox=dict(boxstyle="round", fc="w")) # <--

plt.title('Cost function')
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.xticks(range(1,len(profile)))
plt.grid()    
plt.show()
#pdb.set_trace()