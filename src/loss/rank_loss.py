# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

class Rank_loss():
    # Layer of Efficient Siamese loss function

    def __init__(self):
        self.margin = 5
        self.parasCreated = False
        self.fc1 = []
        self.fc2 = []
        self.fc3 = []

    def _combine(self, feature1, feature2):
        out = tf.concat([feature1, feature2], 0)  #特征拼接
        out = tf.reshape(out, shape=[1,256])

        with tf.variable_scope('combine_fc1', reuse=tf.AUTO_REUSE) as scope:
            fc1w = tf.get_variable('weights', initializer=tf.truncated_normal([256, 64], dtype=tf.float32, stddev=1))
            fc1b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            fc1 = tf.nn.bias_add(tf.matmul(out, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('combine_fc2', reuse=tf.AUTO_REUSE) as scope:
            fc2w = tf.get_variable('weights', initializer=tf.truncated_normal([64, 32], dtype=tf.float32, stddev=1))
            fc2b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[32], dtype=tf.float32))
            fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('combine_fc3', reuse=tf.AUTO_REUSE) as scope:
            fc3w = tf.get_variable('weights', initializer=tf.truncated_normal([32, 1], dtype=tf.float32, stddev=1))
            fc3b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[1], dtype=tf.float32))
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
            #fc3 = tf.nn.sigmoid(fc3)

        return fc3[0], fc1, fc2, fc3


    def _combine2(self, feature1, feature2):
        out = tf.concat([feature1, feature2], 1)  #特征拼接
        out = tf.reshape(out, shape=[-1,256])

        with tf.variable_scope('combine_fc1', reuse=tf.AUTO_REUSE) as scope:
            fc1w = tf.get_variable('weights', initializer=tf.truncated_normal([256, 64], dtype=tf.float32, stddev=1))
            fc1b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            fc1 = tf.nn.bias_add(tf.matmul(out, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('combine_fc2', reuse=tf.AUTO_REUSE) as scope:
            fc2w = tf.get_variable('weights', initializer=tf.truncated_normal([64, 32], dtype=tf.float32, stddev=1))
            fc2b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[32], dtype=tf.float32))
            fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('combine_fc3', reuse=tf.AUTO_REUSE) as scope:
            fc3w = tf.get_variable('weights', initializer=tf.truncated_normal([32, 1], dtype=tf.float32, stddev=1))
            fc3b = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[1], dtype=tf.float32))
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
            #fc3 = tf.nn.sigmoid(fc3)

        return fc3, fc1, fc2, fc3


    def get_rankloss(self, p_hat, batch_size):
        """The forward """
        self.Num = 0
        batch = 1
        level = 5
        dis = 5 #失真类型
        SepSize = batch * level
        self.dis = []
        self.dis2 = []
        self.loss = 0
        for k in range(dis):
            for i in range(SepSize * k, SepSize * (k + 1) - batch):
                for j in range(SepSize * k + int((i - SepSize * k) / batch + 1) * batch, SepSize * (k + 1)):
                    dis, fc1, fc2, fc3 = self._combine(p_hat[i], p_hat[j])
                    dis2, _, _, _ = self._combine(p_hat[j], p_hat[i])
                    self.dis.append(dis)
                    self.dis2.append(dis2)
                    self.fc1.append(fc1)
                    self.fc2.append(fc2)
                    self.fc3.append(fc3)
                    self.Num += 1
        self.dis = tf.cast(tf.reshape(self.dis, [-1]), tf.float32)
        ones = tf.ones(50)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis, labels=ones)
        loss = tf.reduce_mean(self.loss)
        # diff = tf.cast(self.margin, tf.float32) - self.dis
        # self.loss = tf.maximum(0., diff)
        # loss = tf.reduce_mean(self.loss)

        self.dis2 = tf.cast(tf.reshape(self.dis2, [-1]), tf.float32)
        zeros = tf.zeros(50)
        self.loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis2, labels=zeros)
        loss += tf.reduce_mean(self.loss2)
        # diff2 = tf.cast(self.margin, tf.float32) + self.dis2
        # self.loss2 = tf.maximum(0., diff2)
        # loss += tf.reduce_mean(self.loss2)
        return loss


if __name__ == "__main__":
    y_hat = tf.placeholder(tf.float32, [24])
    rank_loss = Rank_loss()
    loss = rank_loss.get_rankloss(y_hat, 24)
    with tf.Session() as sess:
        y = np.random.random([24])
        _loss = sess.run([loss], feed_dict={y_hat: y})
        print(_loss)
