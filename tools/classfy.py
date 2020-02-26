#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from nets.nasnet import nasnet


# config
imgSize = 299
classNum = 5
batchSize = 32
learningRate = 1e-4
isTraining = True
filenamePath = "E:/database/tid2013/mosStd_with_names_4_descend.txt"

# load
filenameList = ["E:/database/tid2013/distorted_images/" + line.rstrip('\n').split(' ')[1] for line in
                   open(filenamePath)]
classDict = {'01' : 0,    '08' : 1,    '10' : 2,   '11' : 3,   '17' : 4}
labelList = [classDict[line.rstrip('\n').split(' ')[1][4:6]] for line in
                   open(filenamePath)]

input_queue = tf.train.slice_input_producer([filenameList, labelList])
label = input_queue[1]
image_raw_data = tf.read_file(input_queue[0])
image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
image = tf.image.resize_image_with_crop_or_pad(image_raw, imgSize, imgSize)
image = tf.cast(image, tf.float32)
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchSize,
                                                      num_threads=32, min_after_dequeue=50, capacity=1000)
label_batch = tf.one_hot(label_batch, classNum)

# model
with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
    logits, end_points = nasnet.build_nasnet_mobile(image_batch, classNum, is_training = isTraining)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = label_batch)
    loss = tf.reduce_mean(loss)


# drive
exclude = ['final_layer', 'aux_7']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=2)
with tf.Session() as sess:
    gd = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss,
                                                                     var_list=[v for v in tf.trainable_variables()])
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../experiments/nasnet_ckpt/model.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10000):
        _, loss_ = sess.run([gd, loss])
        print(loss_)

    saver.save(sess, "../experiments/nasnet_classfy")
