#! /usr/bin/env python
# -*-coding: UTF-8 -*-

import os
import sys
import time
import tensorflow as tf
from flask import (Flask, Markup, escape, make_response, redirect,
                   render_template, request, send_file, session, url_for)
import simplejson as json
from bson import json_util
from flask_bootstrap import Bootstrap
sys.path.append('../')
from tools import test_rank

# initializing a variable of flask
app = Flask(__name__)
# initializing a variable of bootstrap
bootstrap = Bootstrap(app)

BASE_DIR = 'F:/PycharmProjects/rank/framework/'
UPLOAD_FOLDER = 'static/XZL_Download/'#文件下载路径
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg', 'bmp'])#文件允许上传格式
anchor_path = [line.rstrip('\n').split(' ')[1] for line in open("F:/PycharmProjects/rank/framework/static/anchor/tid2013/tid2013.txt")]
anchor_score = [line.rstrip('\n').split(' ')[0] for line in open("F:/PycharmProjects/rank/framework/static/anchor/tid2013/tid2013.txt")]
anchor_index = 0
anchor_num = 21


def allowed_file(filename):  # 通过将文件名分段的方式查询文件格式是否在允许上传格式范围之内
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# when change thread(flask will process request by a multi-thread way), tf will create a new graph,
# it will make some mistake cause the new graph will be empty, so I catch it first and make it global
tf_graph = tf.get_default_graph()
#global tf_graph

def make_json_response(body, status_code=200):
    #print('body:', body)
    resp = make_response(json.dumps(body, default=json_util.default))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp

@app.route('/')
def index():
    return render_template('XZL.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            timepath = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            filepath = os.path.join(BASE_DIR+UPLOAD_FOLDER, timepath)
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            save_name = file.filename
            file.save(os.path.join(filepath, save_name))

            test_rank.calculate_file1(os.path.join(filepath, save_name))

            return make_json_response({
                'result': 'ok',
                'filename': save_name,
                'filepath': timepath,
            })
        else:
            return make_json_response({
                'result': 'fall',
            })

@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    global anchor_index
    global anchor_path
    global anchor_num
    if request.method == 'POST':
        img_dir = request.form['src']
        #img = cv2.imread(img_dir)
        img2_dir = anchor_path[anchor_index]
        img2_name = img2_dir.split('/')[-1].split('.')[0]
        result = test_rank.calculate_file2('F:/PycharmProjects/rank/framework' + img2_dir)
        anchor_index += 1
        if (anchor_index == anchor_num):
            isEnd = 1
            anchor_index = 0
        else:
            isEnd = 0


        return make_json_response({
            'result': 'ok',
            #'score': 100,
            'score' : result,
            'second_img_name':img2_name,
            'second_img_path': img2_dir,
            'extra_text':  anchor_score[anchor_index-1],
            'isEnd': isEnd,
        })


if __name__ == "__main__":
    app.debug = True
    # app.run(host='202.205.18.4', port=8000, debug=True, threaded=True)
    app.run(host='127.0.0.1', port=8888, debug=True, use_reloader=False)#, threaded=True)