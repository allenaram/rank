#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import time
from datetime import datetime

import sys
sys.path.append('../')

from src.datasets.rank_dataloader import Dataset
from src.loss.rank_loss import Rank_loss
from src.net.model import VggNetModel_NewVer
from src.net.model import ResNet50
from src.utils.checkpoint import save
from src.utils.logger import setup_logger

import tensorflow as tf
slim = tf.contrib.slim

isFineTuning = True
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]
db_base = 'E:/database'
ckpt_base = os.path.abspath('..').replace('\\', '/') + "/experiments"
data_root = db_base + "/tid2013/mosStd_with_names_ascend.txt" if isFineTuning \
                                    else db_base + "/tid2013/mos_with_names_new.txt"

modelName = "ResNet50"
modelDict = {
    "VGG16" : VggNetModel_NewVer,
    "ResNet50" : ResNet50,
    "ResNet50_2" : ResNet50
}
ckptPath = {
    "VGG16" : ckpt_base + "/vgg_models/" + 'vgg16_weights.npz',

    "ResNet50" : ckpt_base + "/tid2013_resnet50_hingeLoss/rankiqa/" + 'model.ckpt-9999'if isFineTuning
        else ckpt_base + "/resnet_ckpt/" + 'resnet_v2_50.ckpt'
}



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# specifying default parameters
def process_command_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow RankIQA Training")

    ## Path related arguments
    parser.add_argument('--exp_name', type=str, default="rankiqa", help='experiment name')
    parser.add_argument('--data_dir', type=str, default=db_base, help='the root path of dataset')
    parser.add_argument('--train_list', type=str, default='live_train.txt', help='data list for read image.')
    parser.add_argument('--test_list', type=str, default='live_test.txt', help='data list for read image.')
    parser.add_argument('--ckpt_dir', type=str, default=os.path.abspath('..').replace('\\', '/') + '/experiments',
                        help='the path of ckpt file')
    parser.add_argument('--logs_dir', type=str, default=os.path.abspath('..').replace('\\', '/') + '/experiments',
                        help='the path of tensorboard logs')
    parser.add_argument('--model_path', type=str,
                        default=os.path.abspath('..').replace('\\', '/') + "/experiments/vgg_models/" + 'vgg16_weights.npz')
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
    parser.add_argument('--is_eval', type=str2bool, default=False, help='whether to test.')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--test_step', type=int, default=500)
    parser.add_argument('--summary_step', type=int, default=10)

    ## optimization related arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help='keep neural node')
    parser.add_argument('--iter_max', type=int, default=50000, help='the maxinum of iteration')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

    args = parser.parse_args()
    return args


def train(args):
    global logger
    #print(logger)

    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.create_global_step()

        ## placeholders for training data (224,224,3)
        imgs = tf.placeholder(tf.float32, [None, args.crop_height, args.crop_width, 3])
        dropout_keep_prob = tf.placeholder(tf.float32, [])
        lr = tf.placeholder(tf.float32, [])

        #with tf.name_scope("create_models"):
        model = modelDict[modelName](num_classes=1, dropout_keep_prob=dropout_keep_prob)
        y_hat = model.inference(imgs, True)

        with tf.name_scope("create_loss"):
            rank_loss = Rank_loss()
            loss = rank_loss.get_rankloss(y_hat, args.batch_size)

        exclude = [] if isFineTuning else\
            ["1x1conv1", "1x1conv2", "1x1conv3",
                   "combine_fc1", "combine_fc2", "combine_fc3",
                   'create_optimize','Adam'
            ]
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        saver4read = tf.train.Saver(var_list=variables_to_restore, max_to_keep=2)

        with tf.name_scope("create_optimize"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
            var_list = [v for v in tf.trainable_variables()]
            var_list += [g for g in  tf.global_variables() if 'moving_mean' in g.name]
            var_list += [g for g in tf.global_variables() if 'moving_variance' in g.name]
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=var_list)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('rank_loss', loss)
        # Build the summary Tensor based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        summary_writer = tf.summary.FileWriter(os.path.join(args.logs_dir, 'train').replace('\\', '/'), filename_suffix=args.exp_name)
        summary_test = tf.summary.FileWriter(os.path.join(args.logs_dir, 'test').replace('\\', '/'),filename_suffix=args.exp_name)

        train_data = Dataset(
            {'data_root':data_root,'im_shape':[224,224],'batch_size':45}, isFineTuning)

        test_data = Dataset(
            {'data_root':data_root,'im_shape':[224,224],'batch_size':45}, isFineTuning)

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        model.load_original_weights(sess, ckptPath[modelName], saver4read)

        # global_var = tf.global_variables()
        # var_list = sess.run(global_var)
        start_time = time.time()
        base_lr = args.learning_rate
        for step in range(args.iter_max):

            if (step + 1) % (0.5 * args.iter_max) == 0:
                base_lr = base_lr / 5
            if (step + 1) % (0.8 * args.iter_max) == 0:
                base_lr = base_lr / 5
            # base_lr=(base_lr-base_lr*0.001)/args.iter_max*(args) # other learning rate modify

            image_batch, label_batch = train_data.next_batch()
            dis1_, dis2_, loss_, _, _= sess.run([rank_loss.dis, rank_loss.dis2, loss, optimizer, y_hat], feed_dict={imgs: image_batch, lr: base_lr,
                                                              dropout_keep_prob: args.dropout_keep_prob})


            if (step + 1) % args.summary_step == 0:

                logger.info("step %d/%d,rank loss is %f, time %f,learning rate: %.8f" % (
                step, args.iter_max, loss_, (time.time() - start_time), base_lr))
                summary_str = sess.run(summary_op, feed_dict={imgs: image_batch, lr: base_lr,
                                                              dropout_keep_prob: args.dropout_keep_prob})
                summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()

            if (step + 1) % args.test_step == 0:
                if args.save_ckpt_file:
                    # saver.save(sess, args.checkpoint_dir + 'iteration_' + str(step) + '.ckpt',write_meta_graph=False)
                    save(saver, sess, args.ckpt_dir, step)

                test_epoch_step = len(test_data.scores) // test_data.batch_size + 1
                test_loss = 0
                for _ in range(test_epoch_step):
                    image_batch, label_batch = test_data.next_batch()
                    loss_, _ = sess.run([loss, y_hat], feed_dict={imgs: image_batch, lr: base_lr,
                                                                  dropout_keep_prob: args.dropout_keep_prob})  # bug: test sets do not run sess optimizer.
                    test_loss += loss_
                test_loss /= test_epoch_step
                s = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss)])
                summary_test.add_summary(s, step)
            if step == args.iter_max:
                saver.save(sess, args.ckpt_dir + 'rank_model_final' + '.ckpt', write_meta_graph=False)

        logger.info("Optimization finish!")


# def main():
args = process_command_args()

if args.dataset == 'tid2013':
    args.train_list = 'tid2013_train.txt'
    args.test_list = 'tid2013_test.txt'
elif args.dataset == 'LIVE':
    args.train_list = 'live_train.txt'
    args.test_list = 'live_test.txt'
elif args.dataset == 'CSIQ':
    args.train_list = 'csiq_train.txt'
    args.test_list = 'csiq_test.txt'
else:
    logger.info("datasets is not in LIVE, CSIQ, tid2013")

output_dir = os.path.join(args.ckpt_dir, args.dataset).replace('\\', '/')
args.data_dir = os.path.join(args.data_dir, args.dataset).replace('\\', '/')
args.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, args.exp_name).replace('\\', '/')
args.logs_dir = os.path.join(args.logs_dir, args.dataset, "logs").replace('\\', '/')

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

logger = setup_logger("TF_rank_training", output_dir, "train_rank_")
logger.info(args)


train(args)


# if __name__ == "__main__":
#     global logger
#     main()
