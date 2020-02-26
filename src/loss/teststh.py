import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import random

# dict = {}
# f = open('E:/database/tid2013/mosStd_with_names.txt')
# lines = f.readlines()
#
# for line in lines:
#     score, name = line.split('\t')
#     name = name[1: -2]
#     dict[name] = score
#
#
# scoresAndNames = []
# names = []
# for dis in ['01', '08', '10', '11', '17']:
#     for level in range(5):
#         for pic in range(25):
#             name = 'i' + (str(pic+1) if (pic+1)>9 else '0'+str(pic+1)) + '_' + dis + '_' + str(level+1) + '.bmp'
#             scoresAndNames.append(dict[name] + ' ' + name + '\n')
#
#
# f = open('E:/database/tid2013/newnewnew.txt', 'w')
# f.writelines(scoresAndNames)

sess = tf.Session()
a = tf.constant([1,2,2,0,4])
b = tf.one_hot(a, 5)
print(sess.run(b))
