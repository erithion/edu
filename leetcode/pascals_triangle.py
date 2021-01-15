import numpy as np
import pdb

N = 1000
_1 = np.array([1])
R = [_1, np.concatenate([_1, _1])]
for i in range(2, N):
    sum_x = np.eye(i, i - 1) + np.eye(i, i - 1, -1)
    R.append(np.concatenate([_1, R[i - 1].dot(sum_x), _1]))
pdb.set_trace()