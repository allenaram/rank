#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created predict.py by rjw at 19-1-11 in WHU.
"""

import os
import tensorflow as tf
import numpy as np
from src.net.model import VggNetModel_NewVer
from src.net.model import ResNet50
from src.loss.rank_loss import Rank_loss
import matplotlib.pyplot as plt

modelName = "VGG16"
modelDict = {
    "VGG16" : VggNetModel_NewVer,
    "ResNet50" : ResNet50
}
ckptPath = {
    "VGG16" : "../experiments/tid2013_vgg16_hingeLoss/rankiqa/model.ckpt-9999",
    "ResNet50" : "../experiments/tid2013_resnet50_hingeLoss/rankiqa/model.ckpt-9999"
}

image = tf.placeholder(tf.float32,[2,224,224,3])
dropout_keep_prob = tf.placeholder(tf.float32)

model = modelDict[modelName](num_classes=1, dropout_keep_prob=dropout_keep_prob)
combine = Rank_loss()

y_hat = model.inference(image, False)
#y_hat = tf.reshape(y_hat, [-1, ])
rank, _, _, _ = combine._combine(y_hat[0], y_hat[1])

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, ckptPath[modelName])


def get_pic(filename, sess):
    image_raw_data = tf.read_file(filename)
    image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
    img = tf.image.resize_images(image_raw, (224, 224))
    # img = (tf.cast(img, tf.float32) - 127.5) / 127.5
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, dim=0)
    input = sess.run(img)
    # input = img3.eval()
    return input


def calculate_file1(filename1):
    global image1
    #filename1 = r"E:\database\tid2013\distorted_images\i01_01_4.bmp"
    image1 = get_pic(filename1, sess)

def calculate_file2(filename2):
    global image1
    #filename2 = r"E:\database\tid2013\distorted_images\i01_01_1.bmp"
    # image_raw_data = tf.gfile.FastGFile(filename, 'r').read()
    image2 = get_pic(filename2, sess)
    inputs = np.concatenate([image1, image2], axis=0)
    flag = sess.run(rank, feed_dict={image: inputs})
    return 1 if(flag>0) else -1





def get_pic_old_ver(filename, sess):
    image_raw_data = tf.read_file(filename)
    image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
    img = tf.image.resize_images(image_raw, (224, 224))
    # img = (tf.cast(img, tf.float32) - 127.5) / 127.5
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, dim=0)
    input = sess.run(img)
    # input = img3.eval()
    return input

def calculate_file1_oldver(filename1):
    global score1
    #filename1 = r"E:\database\tid2013\distorted_images\i01_01_4.bmp"
    image1 = get_pic(filename1, sess)
    score1 = sess.run(y_hat, feed_dict={image: image1})[0]

def calculate_file2_oldver(filename2):
    global score1
    #filename2 = r"E:\database\tid2013\distorted_images\i01_01_1.bmp"
    # image_raw_data = tf.gfile.FastGFile(filename, 'r').read()
    image2 = get_pic(filename2, sess)
    score2 = sess.run(y_hat, feed_dict={image: image2})[0]
    return 1 if(score1 > score2) else -1


if __name__ == "__main__":
    print(evaluate())