from operator import indexOf
import numpy as np
from numpy.core.numeric import tensordot

np.random.seed(41)
t = np.arange(20)
t = t.reshape(20, 1)
np.random.shuffle(t)
print(t)
print(np.where(t == [3]))
