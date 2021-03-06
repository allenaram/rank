"""
Derived from: https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
              https://github.com/dgurkaynak/tensorflow-cnn-finetune/blob/master/vggnet/model.py
"""
import tensorflow as tf
import numpy as np
from nets import vgg
from nets import resnet_v2
from nets import inception
from nets import inception_v3
from nets.nasnet import nasnet
slim = tf.contrib.slim


class VggNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.feature =None

    def inference(self, x, training=False):
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope.name)

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope.name)

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope.name)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope.name)

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope.name)

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope.name)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope.name)

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope.name)

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope.name)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope.name)

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope.name)

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope.name)

        # pool5
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
        self.feature = pool5

        # fc6
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', initializer=tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1))
            fc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            fc6 = tf.nn.relu(fc6l)

            if training:
                fc6 = tf.nn.dropout(fc6, self.dropout_keep_prob)

        # fc7
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1))
            fc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
            fc7 = tf.nn.relu(fc7l)

            if training:
                fc7 = tf.nn.dropout(fc7, self.dropout_keep_prob)

        # fc8
        with tf.variable_scope('fc8') as scope:
            fc8w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, self.num_classes], dtype=tf.float32, stddev=1e-1))
            fc8b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.num_classes], dtype=tf.float32))
            self.score = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

        return self.score

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

    def load_original_weights(self, session,vgg_models_path, skip_layers=[]):
        weights = np.load(vgg_models_path)
        keys = sorted(weights.keys())

        for i, name in enumerate(keys):
            parts = name.split('_')
            layer = '_'.join(parts[:-1])

            # if layer in skip_layers:
            #     continue

            if layer == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope(layer, reuse=True):
                if parts[-1] == 'W':
                    var = tf.get_variable('weights')
                    session.run(var.assign(weights[name]))
                elif parts[-1] == 'b':
                    var = tf.get_variable('biases')
                    session.run(var.assign(weights[name]))


class VggNetModel_NewVer(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, isFineTuning = False):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.feature =None
        self.isFineTuning = isFineTuning

    def inference(self, x, training=False):
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope.name)

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope.name)

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope.name)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope.name)

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope.name)

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope.name)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope.name)

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope.name)

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope.name)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope.name)

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope.name)

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope.name)

        # pool5
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
        self.feature = pool5

        # conv6
        with tf.variable_scope('conv6') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[4096], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv1
        with tf.variable_scope('1x1conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 4096, 1024], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[1024], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_1 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv2
        with tf.variable_scope('1x1conv2') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 1024, 512], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv1x1_1, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_2 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv3
        with tf.variable_scope('1x1conv3') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 512, 128], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv1x1_2, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_3 = tf.nn.relu(out, name=scope.name)
            conv1x1_3 = tf.reshape(conv1x1_3, [-1, 128])

        # fc6
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', initializer=tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1))
            fc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            fc6 = tf.nn.relu(fc6l)

            if training:
                fc6 = tf.nn.dropout(fc6, self.dropout_keep_prob)

        # fc7
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1))
            fc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
            fc7 = tf.nn.relu(fc7l)

            if training:
                fc7 = tf.nn.dropout(fc7, self.dropout_keep_prob)

        # fc8
        with tf.variable_scope('fc8') as scope:
            fc8w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, self.num_classes], dtype=tf.float32, stddev=1e-1))
            fc8b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.num_classes], dtype=tf.float32))
            self.score = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

        return conv1x1_3

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

    def load_original_weights(self, session, vgg_models_path, saver, skip_layers=[]):
        if not self.isFineTuning:
            weights = np.load(vgg_models_path)
            keys = sorted(weights.keys())

            for i, name in enumerate(keys):
                parts = name.split('_')
                layer = '_'.join(parts[:-1])

                # if layer in skip_layers:
                #     continue

                if layer == 'fc8' and self.num_classes != 1000:
                    continue

                with tf.variable_scope(layer, reuse=True):
                    if parts[-1] == 'W':
                        var = tf.get_variable('weights')
                        session.run(var.assign(weights[name]))
                    elif parts[-1] == 'b':
                        var = tf.get_variable('biases')
                        session.run(var.assign(weights[name]))
        else:
            saver.restore(session, vgg_models_path)


class Vgg16(object):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, isFineTuning = False):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

    def inference(self, x, training=False):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(x, 1000, is_training=True)

        with tf.variable_scope('1x1conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 4096, 512], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(end_points["vgg_16/fc6"], kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_1 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv2
        with tf.variable_scope('1x1conv2') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 512, 128], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv1x1_1, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope.name)
            out = tf.reshape(out, [-1, 128])

        return out

    def load_original_weights(self, session,ckpt_path, saver, skip_layers=[]):
        saver.restore(session, ckpt_path)


class ResNet50(object):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, isFineTuning = False):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob


    def inference(self, x, training=False):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            _, end_points = resnet_v2.resnet_v2_50(x, 1001, is_training=True)

        with tf.variable_scope('1x1conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 2048, 1024], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(end_points["global_pool"], kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[1024], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_1 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv2
        with tf.variable_scope('1x1conv2') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 1024, 512], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv1x1_1, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_2 = tf.nn.relu(out, name=scope.name)

        # 1x1 conv3
        with tf.variable_scope('1x1conv3') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 512, 128], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(conv1x1_2, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1x1_3 = tf.nn.relu(out, name=scope.name)
            conv1x1_3 = tf.reshape(conv1x1_3, [-1, 128])

        return conv1x1_3

    def load_original_weights(self, session,ckpt_path, saver, skip_layers=[]):
        saver.restore(session, ckpt_path)


class InceptionV3(object):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, isFineTuning = False):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.isFineTuning = isFineTuning


    def inference(self, x, training = False):
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(x, 1001, training, create_aux_logits=False,)

        with tf.variable_scope('1x1conv1') as scope:
            kernel = tf.get_variable('weights',
                                     initializer=tf.truncated_normal([1, 1, 2048, 128], dtype=tf.float32,
                                                                     stddev=1e-1))
            conv = tf.nn.conv2d(end_points["AvgPool_1a"], kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            out = tf.reshape(out, [-1, 128])

        return out

    def load_original_weights(self, session,ckpt_path, saver, skip_layers=[]):
        saver.restore(session, ckpt_path)


class NasNet_Mobile(object):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, isFineTuning = False):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.isFineTuning = isFineTuning


    def inference(self, x, training = False):
        with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
            logits, end_points = nasnet.build_nasnet_mobile(x, 1001, is_training = training)

        with tf.variable_scope('finalFC') as scope:
            w = tf.get_variable('weights',
                                   initializer=tf.truncated_normal([1056, 128], dtype=tf.float32,
                                                                   stddev=0.1))
            b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(tf.matmul(end_points["global_pool"], w), b)

        return out#tf.nn.softmax(end_points["AuxLogits"])

    def load_original_weights(self, session,ckpt_path, saver, skip_layers=[]):
        saver.restore(session, ckpt_path)