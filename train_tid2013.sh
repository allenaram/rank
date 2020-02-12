#!/usr/bin/env bash
nvidia-smi

PYTHON="python"

#dataset config
DATASET="tid2013"

#training setting
GPU="0"
LEARN_RATE=6e-5
END_LR=6e-6

WORK_DIR='/home/rjw/desktop/graduation_project/TF_RankIQA'

$PYTHON "${WORK_DIR}"/tools/train_iqa_v2.py  \
      --dataset $DATASET    \
      --learning_rate $LEARN_RATE    \
      --start_lr $END_LR
