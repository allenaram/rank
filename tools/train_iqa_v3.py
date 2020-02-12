#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created train_iqa_v3 by rjw at 19-3-10 in WHU.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# print(sys.path)

from src.datasets.nima_iqa_dataloader import train_generator, val_generator
from src.loss.EMD_loss import _emd
from src.metrics.srocc import evaluate_metric, scores_stats
from src.net.model import VggNetModel
from src.net.vgg16_model import fully_connection
from src.utils.checkpoint import save, load, __load__
from src.utils.logger import setup_logger
from src.utils.max_entropy import get_max_entropy_distribution


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


experiment_name = os.path.splitext(__file__.split('/')[-1])[0]
BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset'


# specifying default parameters
def process_command_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow RankIQA Training")

    ## Path related arguments
    parser.add_argument('--exp_name', type=str, default="finetuneiqav3", help='experiment name')
    parser.add_argument('--data_dir', type=str, default=BASE_PATH, help='the root path of dataset')
    parser.add_argument('--train_list', type=str, default=None,
                        help='train data list for read image.')
    parser.add_argument('--test_list', type=str, default=None,
                        help='test data list for read image.')
    parser.add_argument('--ckpt_dir', type=str, default=rootPath + '/experiments',
                        help='the path of ckpt file')
    parser.add_argument('--logs_dir', type=str, default=rootPath + '/experiments',
                        help='the path of tensorboard logs')
    parser.add_argument('--pretrain_models_path', type=str,
                        default=rootPath + "/experiments/LIVE/rankiqa/" + 'model.ckpt-6999_ft')

    ## models retated argumentss
    parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                        help="whether to save trained checkpoint file ")

    ## dataset related arguments
    parser.add_argument('--dataset', default='tid2013', type=str, choices=["LIVE", "CSIQ", "tid2013"],
                        help='datset choice')
    parser.add_argument('--crop_width', type=int, default=224, help='train patch width')
    parser.add_argument('--crop_height', type=int, default=224, help='train patch height')

    ## train related arguments
    parser.add_argument('--is_training', type=str2bool, default=True, help='whether to train or test.')
    parser.add_argument('--is_eval', type=str2bool, default=True, help='whether to test.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_step', type=int, default=500)
    parser.add_argument('--summary_step', type=int, default=10)

    ## optimization related arguments
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='init learning rate')
    parser.add_argument('--start_lr', type=float, default=1e-6, help='init learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help='keep neural node')
    parser.add_argument('--iter_max', type=int, default=9000, help='the maxinum of iteration')
    parser.add_argument('--epoch', type=int, default=64)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

    args = parser.parse_args()
    return args


def get_image_list(args):
    train_image_paths = []
    train_scores = []
    train_mean = []
    f = open(args.train_list, 'r')
    for line in f:
        image_path, image_score, _ = line.strip("\n").split()
        train_image_paths.append(os.path.join(BASE_PATH, args.dataset, image_path))
        train_mean.append(image_score)
        score_10 = get_max_entropy_distribution(float(image_score))
        train_scores.append(score_10.tolist())
    f.close()
    train_image_paths = np.array(train_image_paths)
    train_scores = np.array(train_scores, dtype='float32')
    train_mean = np.array(train_mean, dtype="float32")

    test_image_paths = []
    test_scores = []
    test_mean = []
    f = open(args.test_list, 'r')
    for line in f:
        image_path, image_score, _ = line.strip("\n").split()
        test_image_paths.append(os.path.join(BASE_PATH, args.dataset, image_path))
        test_mean.append(image_score)
        score_10 = get_max_entropy_distribution(float(image_score))
        test_scores.append(score_10)
    f.close()
    test_image_paths = np.array(test_image_paths)
    test_scores = np.array(test_scores, dtype='float32')
    test_mean = np.array(test_mean, dtype="float32")

    return train_image_paths, train_scores, train_mean, test_image_paths, test_scores, test_mean


def train(args):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.create_global_step()

        # # placeholders for training data
        images = tf.placeholder(tf.float32, [None, args.crop_height, args.crop_width, 3])
        scores = tf.placeholder(tf.float32, [None, 10])
        dropout_keep_prob = tf.placeholder(tf.float32, [])
        lr = tf.placeholder(tf.float32, [])

        with tf.name_scope("create_models"):
            model = VggNetModel(num_classes=1, dropout_keep_prob=dropout_keep_prob)
            model.inference(images, True)
            x = fully_connection(model.feature, 128, 0.5)
            y_hat = tf.nn.softmax(x)

        with tf.name_scope("create_loss"):
            emd_loss_out = _emd(y_hat, scores)
            mean_hat = scores_stats(y_hat)

        with tf.name_scope("create_optimize"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss) # not converge ??
            var_list = [v for v in tf.trainable_variables()]
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(emd_loss_out, var_list=var_list)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        # 得到该网络中，所有可以加载的参数; 用var_list = tf.contrib.framework.get_variables(scope_name)获取指定scope_name下的变量，
        variables = tf.contrib.framework.get_variables_to_restore()
        # 删除output层中的参数
        variables_to_resotre = [v for v in variables if v.name.startswith("conv")]
        # 构建这部分参数的
        pre_saver = tf.train.Saver(variables_to_resotre)

        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('reg_loss', emd_loss_out)
        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        summary_writer = tf.summary.FileWriter(
            os.path.join(args.logs_dir, 'train/{}-{}'.format(args.exp_name, timestamp)), filename_suffix=args.exp_name)
        summary_test = tf.summary.FileWriter(os.path.join(args.logs_dir, 'test/{}-{}'.format(args.exp_name, timestamp)),
                                             filename_suffix=args.exp_name)

        train_image_paths, train_scores, train_mean, test_image_paths, test_scores, test_mean = get_image_list(args)
        train_loader = train_generator(train_image_paths, train_scores, train_mean)
        train_num_batchs = len(train_image_paths) // args.batch_size + 1

        test_loader = val_generator(test_image_paths, test_scores, test_mean)
        test_num_batchs = len(test_image_paths) // args.batch_size + 1

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        counter = 0
        if ckpt and ckpt.model_checkpoint_path:
            counter = __load__(saver, sess, args.ckpt_dir)
        else:
            load(pre_saver, sess, args.pretrain_models_path)

        start_time = time.time()
        start_step = counter  # if counter is not None else 0

        base_lr = args.learning_rate
        for step, (batch_images, batch_targets, _) in enumerate(train_loader, start_step):

            if step <= 500:
                base_lr = args.start_lr + (args.learning_rate - args.start_lr) * step / float(500)
            else:
                if (step + 1) % (0.5 * args.iter_max) == 0:
                    base_lr = base_lr / 5
                if (step + 1) % (0.8 * args.iter_max) == 0:
                    base_lr = base_lr / 5
            # base_lr=(base_lr-base_lr*0.001)/args.iter_max*(args) # other learning rate modify

            loss_, _ = sess.run([emd_loss_out, optimizer],
                                feed_dict={images: batch_images, scores: batch_targets, lr: base_lr,
                                           dropout_keep_prob: args.dropout_keep_prob})

            if (step + 1) % args.summary_step == 0:
                # logger.info("targets labels is : {}".format(targets))
                # logger.info("predict lables is : {}".format(y_hat_))

                logger.info("step %d/%d,reg loss is %f, time %f,learning rate: %.8f" % (
                    step, args.iter_max, loss_, (time.time() - start_time), base_lr))
                summary_str = sess.run(summary_op, feed_dict={images: batch_images, scores: batch_targets, lr: base_lr,
                                                              dropout_keep_prob: args.dropout_keep_prob})
                summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()

            if (step + 1) % args.test_step == 0:
                if args.save_ckpt_file:
                    # saver.save(sess, args.checkpoint_dir + 'iteration_' + str(step) + '.ckpt',write_meta_graph=False)
                    save(saver, sess, args.ckpt_dir, step)
                test_loss = 0
                mean_set = np.array([])
                lables_set = np.array([])
                # for step, (images, targets) in enumerate(test_loader):
                for i in range(test_num_batchs):
                    batch_images, batch_targets, batch_mean = next(test_loader)
                    loss_, y_hat_, mean_hat_ = sess.run([emd_loss_out, y_hat, mean_hat],
                                                        feed_dict={images: batch_images, scores: batch_targets,
                                                                   lr: base_lr,
                                                                   dropout_keep_prob: args.dropout_keep_prob})
                    test_loss += loss_
                    mean_set = np.append(mean_set, mean_hat_)
                    lables_set = np.append(lables_set, batch_mean)
                    logger.info('test_loader step/len(test_loader) :{}/{}'.format(i, test_num_batchs))

                print(type(mean_set), type(lables_set))
                logger.info("mean_set:{}, lables_set:{}.".format(mean_set.shape, lables_set.shape))
                srocc, krocc, plcc, rmse, mse = evaluate_metric(lables_set, mean_set)
                test_loss /= test_num_batchs
                logger.info(
                    "SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f\t test loss: %.3f\n" % (
                        srocc, krocc, plcc, rmse, mse, test_loss))
                s1 = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss)])
                s2 = tf.Summary(value=[tf.Summary.Value(tag='test_srocc', simple_value=srocc)])
                summary_test.add_summary(s1, step)
                summary_test.add_summary(s2, step)

            if step == args.iter_max:
                saver.save(sess, args.ckpt_dir + '/final_model_' + timestamp + '.ckpt', write_meta_graph=False)
                logger.info('save train_iqa final models max_iter: {}...'.format(args.iter_max))
                break

        logger.info("Optimization finish!")


def main():
    args = process_command_args()

    if args.dataset == 'tid2013':
        args.train_list = os.path.join(BASE_PATH, args.dataset, "tid2013_train.txt")
        args.test_list = os.path.join(BASE_PATH, args.dataset, "tid2013_test.txt")
    elif args.dataset == 'LIVE':
        args.train_list = os.path.join(BASE_PATH, args.dataset, "live_train.txt")
        args.test_list = os.path.join(BASE_PATH, args.dataset, "live_test.txt")
    elif args.dataset == 'CSIQ':
        args.train_list = os.path.join(BASE_PATH, args.dataset, "csiq_train.txt")
        args.test_list = os.path.join(BASE_PATH, args.dataset, "csiq_test.txt")
    else:
        logger.info("datasets is not in LIVE, CSIQ, tid2013")

    output_dir = os.path.join(args.ckpt_dir, args.dataset)
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, args.exp_name)
    args.logs_dir = os.path.join(args.logs_dir, args.dataset, "logs")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    global logger
    logger = setup_logger("TF_IQA_" + args.dataset + "_training", output_dir, "train_iqav3")
    logger.info(args)

    train(args)


if __name__ == "__main__":
    main()
