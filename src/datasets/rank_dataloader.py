# -*- coding:utf-8 -*-
import cv2
import numpy as np
import random

class Dataset():

    def __init__(self, param_str, isFineTuning = False):

        # === 读取输入参数 ===
        # 字典输入.
        self.param_str = param_str
        self.batch_size = self.param_str['batch_size']
        self.data_root = self.param_str['data_root']

        # 获取图像索引列表.
        filename = [line.rstrip('\n') for line in open(self.data_root)]
        self._roidb = []
        self.scores = []
        for i in filename:
            self.scores.append(float(i.split()[0]))
            self._roidb.append(i.split()[1].lower())

        self.num = 0
        self.isFineTuning = isFineTuning

    def _get_next_minibatch_inds(self):
        # mini_batch = level * dis_mini * batch
        db_inds = []
        dis = 9  # 失真类型数量
        batch = 1  # 一个minibatch中原图的数量
        level = 5  # 失真等级数量
        Num = len(self.scores) / dis / level # 原图数量
        for k in range(dis):
            for i in range(level):
                temp = self.num
                for j in range(batch):
                    db_inds.append(len(self.scores) / dis * k + i * Num + temp) # k块，i行 ，temp列
                    temp = temp + 1
        self.num = self.num + batch
        if Num - self.num < batch:
            self.num = 0
        db_inds = np.asarray(db_inds)
        return db_inds

    def _get_next_minibatch_inds_ft(self):
        # mini_batch = level * dis_mini * batch
        sampleTarget = [i for i in range(1125)]
        db_inds = []
        dis = 9  # 失真类型数量
        batch = 1  # 一个minibatch中原图的数量
        level = 5  # 失真等级数量
        Num = len(self.scores) / dis / level  # 原图数量
        for k in range(dis):
            idxList = random.sample(sampleTarget, level)
            idxList.sort()
            idxList.reverse()
            db_inds += idxList
        self.num = self.num + batch
        if Num - self.num < batch:
            self.num = 0
        db_inds = np.asarray(db_inds)
        return db_inds

    def get_minibatch(self, minibatch_db):
        # 给定一个roidb，构建一个从中取样的小批量
        jobs = [preprocess('E:/database/tid2013/distorted_images/'+i) for i in minibatch_db]
        index = 0
        images_train = np.zeros([self.batch_size, 224, 224 ,3], np.float32)
        for index_job in range(len(jobs)):
            images_train[index] = jobs[index_job]
            index += 1
        #images_train = random.sample(jobs, self.batch_size)
        blobs = {'data': images_train}
        return blobs

    def next_batch(self):
        # 获取blob
        db_inds = self._get_next_minibatch_inds_ft() if self.isFineTuning else self._get_next_minibatch_inds()
        minibatch_db = []
        for i in range(len(db_inds)):
            minibatch_db.append(self._roidb[int(db_inds[i])])
        scores = []
        for i in range(len(db_inds)):
            scores.append(self.scores[int(db_inds[i])])
        blobs = self.get_minibatch(minibatch_db)
        blobs['label'] = np.asarray(scores)
        #print(minibatch_db)
        #print(blobs['label'])
        return blobs['data'],blobs['label']

def preprocess(data):
    sp = 224
    im = np.asarray(cv2.imread(data))
    x = im.shape[0]
    y = im.shape[1]
    x_p = np.random.randint(x - sp, size=1)[0]
    y_p = np.random.randint(y - sp, size=1)[0]
    images = im[x_p:x_p + sp, y_p:y_p + sp, :]
    return images

if __name__=="__main__":
    data_root="../../../tid2013/mos_with_names.txt"
    param_str={'data_root':data_root,'im_shape':[224,224],'batch_size': 120}
    rank_data = Dataset(param_str)

    for i in range(10):
        image_batch, label_batch = rank_data.next_batch()
        print(image_batch.shape,label_batch.shape)

