import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

a = [[1,2,3]]
b = [[4,5,6]]
c = np.concatenate([a,b],axis=0)
print(c)