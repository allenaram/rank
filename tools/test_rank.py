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
from src.net.model import InceptionV3
from src.net.model import NasNet_Mobile
from src.loss.rank_loss import Rank_loss
import matplotlib.pyplot as plt

modelName = "test"
modelDict = {
    "VGG16" : VggNetModel_NewVer,
    "ResNet50" : ResNet50,
    "InceptionV3" : InceptionV3,
    "NasNet" : NasNet_Mobile,
    "test" : InceptionV3
}
ckptPath = {
    "VGG16" : "../experiments/tid2013_vgg_hingeLoss/rankiqa/model.ckpt-9999",
    "ResNet50" : "../experiments/tid2013_resnet50_hingeLoss_ft/rankiqa/model.ckpt-49999",
    "InceptionV3" : "../experiments/tid2013_inceptionV3_BiLoss/rankiqa/model.ckpt-9999",
    "NasNet" : "../experiments/tid2013_inceptionV3_BiLoss/rankiqa/model.ckpt-9999",
    "test" : "../experiments/tid2013/rankiqa/model.ckpt-49999",
}


img_size = {
    "VGG16" : 224,
    "ResNet50" : 224,
    "InceptionV3" : 299,
    "NasNet" : 299,
    "test" : 299,
}

image = tf.placeholder(tf.float32,[None,img_size[modelName],img_size[modelName],3])
dropout_keep_prob = tf.placeholder(tf.float32)

model = modelDict[modelName](num_classes=1, dropout_keep_prob=dropout_keep_prob)
combine = Rank_loss()

y_hat = model.inference(image, True)
#y_hat = tf.reshape(y_hat, [-1, ])
rank, _, _, _ = combine._combine(y_hat[0], y_hat[1])



def get_pic(filename, sess):
    global img_size
    image_raw_data = tf.read_file(filename)
    image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
    img = tf.image.resize_images(image_raw, (img_size[modelName], img_size[modelName]))
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
    # anchor_path = ["../framework" + line.rstrip('\n').split(' ')[1] for line in
    #                open("F:/PycharmProjects/rank/framework/static/anchor/tid2013/tid2013.txt")]
    # num_of_anchor = 21
    # num_of_interval = num_of_anchor - 1
    # interval_down_threshold = [i*(100/num_of_interval) for i in range(num_of_interval)]
    # counter = np.zeros(num_of_interval)
    #
    # #特征提取
    # inputs = []
    # calculate_file1("E:/database/tid2013/distorted_images/i01_04_1.bmp")
    # inputs.append(image1[0])
    # for i in range (num_of_anchor):
    #     calculate_file1(anchor_path[i])
    #     inputs.append(image1[0])
    # inputs = np.array(inputs)
    # features_ = sess.run(y_hat, feed_dict={image: inputs})
    # features = tf.convert_to_tensor(features_)
    # #排序
    # for i in range(num_of_anchor):
    #     idx = i + 1
    #     flag, _, _, _ = combine._combine(features[0], features[idx])
    #     flag = tf.nn.sigmoid(flag)
    #     flag_ = sess.run(flag)
    #     if flag_ >= 0.5:
    #         for j in range(num_of_interval - i):
    #             counter[i+j] += 1
    #     else:
    #         for j in range(i):
    #             counter[j] += 1
    #
    # maxIdxList = []
    # curMaxIdx = 0
    # curMax = 0
    # for i in range(num_of_interval):
    #     if (curMax < counter[i]):
    #         curMax = counter[i]
    #         curMaxIdx = i
    #
    # print(curMaxIdx *100/num_of_interval + 100/num_of_interval/2)
    # print(111)


    # features = []
    # for i in range(5):
    #     print('特征提取： ' + str(i))
    #     calculate_file1(anchor_path[i])
    #     features_ = sess.run(y_hat, feed_dict={image: image1})
    #     features.append(tf.convert_to_tensor(features_))


    # for i in range(625):
    #     for j in range(625-i-1):
    #         k = j+1
    #         flag, _, _, _ = combine._combine(features[i], features[k])
    #         flag = tf.nn.sigmoid(flag)
    #         flag_ = sess.run(flag)
    #         all+=1
    #         if flag_ >=0.5:
    #             cnt += 1
    #         else:
    #             f.write(str(scores[i] - scores[k]))
    #             f.write('\n')
    #         print("比较： pic" +  str(i) + " vs pic" + str(k) + "      目前准确率： " + str(cnt) + '|' + str(all))
    left = tf.placeholder(tf.float32, [625-1, 128])
    right = tf.placeholder(tf.float32, [625-1, 128])
        # right = features[i+1]
        # for j in range(625):
        #     left = features[i] if j == 0 else tf.concat([left, features[i]], 0)
        #     right = features[i+1] if j == 0 else tf.concat([right, features[j+i+1]], 0)
    flag, _, _, _ = combine._combine2(left, right)
    flag = tf.nn.sigmoid(flag)
        # flag_ = sess.run(flag)
        # cur_cnt = 5-i+1
        # all_cnt += cur_cnt
        # cur_correct = 0
        # for j in range(5-i-1):
        #     if flag_[j] >=0.5:
        #         cur_correct+=1
        #     else:
        #         f.write(str(scores[i]-scores[j+i+1])+'\n')
        # all_correct += cur_correct
        # print("pic" + str(i) + "    " + str(cur_correct) + '|' + str(cur_cnt) + "     累积准确率： " + str(all_correct/all_cnt))

    filename = tf.placeholder(tf.string)
    image_raw_data = tf.read_file(filename)
    image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
    img = tf.image.resize_image_with_crop_or_pad(image_raw, img_size[modelName], img_size[modelName])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, dim=0)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, ckptPath[modelName])




    ####  处理数据
    anchor_path = ["E:/database/tid2013/distorted_images/" + line.rstrip('\n').split(' ')[1] for line in
                   open("E:/database/tid2013/mosStd_with_names_4_descend.txt")]
    scores = [float(line.rstrip('\n').split(' ')[0]) for line in
              open("E:/database/tid2013/mosStd_with_names_4_descend.txt")]

    features = []
    for name in anchor_path:
        img_ = sess.run(img, feed_dict={filename : name})
        features.append(sess.run(y_hat, feed_dict={image: img_}))

    f = open("F:/PycharmProjects/rank/experiments/rank_wrong_list.txt", 'a')
    all_cnt = 0
    cur_cnt = 624
    all_correct = 0
    for i in range(625):
        cur_correct = 0
        left_ = np.array([])
        right_ = np.array([])
        left_scores = []
        right_scores = []
        for j in range(625):
            if i == j:
                continue
            left_ = features[i] if not left_.any() else np.concatenate([left_, features[i]], 0)
            right_ = features[j] if not right_.any() else np.concatenate([right_, features[j]], 0)
            left_scores.append(scores[i])
            right_scores.append(scores[j])
        flag_ = sess.run(flag, feed_dict={left:left_, right:right_})
        for j in range(624):
            if (left_scores[j] - right_scores[j]) * (flag_[j]-0.5) >=0 :
                cur_correct+=1
            else:
                f.write(str(scores[i] - scores[j]) + '\n')
        all_correct += cur_correct
        all_cnt += 624
        print("pic" + str(i) + "    " + str(cur_correct) + '|' + str(cur_cnt) + "     累积准确率： " + str(all_correct/all_cnt))

    f.close()

