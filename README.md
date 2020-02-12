# TF_RankIQA

- Native Tensorflow API Implement of [RankIQA: Learning from Rankings for No-reference Image Quality Assessment](https://arxiv.org/abs/1707.08347). 

```
.
├── data -> /home/**/desktop/RankIQA/data
├── datas
│   ├── CLIVE.py
│   ├── CSIQ.py
│   ├── generate_image_list
│   ├── __init__.py
│   ├── LIVE.py
│   └── TID2013.py
├── debug.txt
├── demo
│   ├── img
│   └── predict.py
├── experiments
│   ├── CLIVE
│   ├── LIVE
│   ├── tid2013
│   └── vgg_models
├── README.md
├── src
│   ├── datasets
│   ├── __init__.py
│   ├── loss
│   ├── metrics
│   ├── net
│   └── utils
├── tools
│   ├── evaluate.py
│   ├── train_clive.py
│   ├── train_iqa.py
│   ├── train_iqa_v2.py
│   ├── train_iqa_v3.py
│   └── train_rank.py
├── train_live.sh
└── train_tid2013.sh


```

## Get Started

- frist you should make soft link: `TF_RankIQA/data -> RankIQA/data/` for data prepare.
    - data dirctory for train rank data
    - datas dirctory for train fintune iqa data(because i find orignal author finetune datasets txt do not strict)
- then you should download the pretrain vgg models from [vgg16_weights.npz](http://www.cs.toronto.edu/~frossard/post/vgg16/) and put it into `TF_RankIQA/experiments/vgg_models/`
- final make sure you have this env: `ubuntu16.04+tensorflow1.8+cuda9.0`

## Run the code

- train for rank: `tools/train_rank.py`
- train for iqa: offical finetune datasets :`tools/train_iqa.py` ;my new finetune datasets: `tools/train_iqa_v2.py`; add nima emd loss :`tools/train_iqa_v3.py` ;or you can run `train_live.sh` or `train_tid2013.sh` ;
- test for iqa: `tools/evaluate.py`
- predict demo: `demo/predict.py`

## Result for experiments

- you can see the tensorboard logs in :`experiments/datasets/logs`
- save train models in :`experiments/datasets/experiments_name/`
- train/test log txt file in :`experiments/datasets/`

## Up coming

- i am finish the overall framwork, but i test srocc/plcc in live datset is not reach paper result.
- next i will going to optimization this project.
- final maybe you will find some bug in the code or you have some advice, please make issue ,thanks!
- last but not least thanks for the offical code : [RankIQA](https://github.com/xialeiliu/RankIQA)


