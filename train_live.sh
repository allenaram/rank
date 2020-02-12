#!/usr/bin/env bash

PYTHON="python"

#dataset config
DATASET="LIVE"

#training setting
GPU="0"
LEARN_RATE=5e-5

$PYTHON -u tools/train_iqa_v2.py  \
      --dataset $DATASET    \
      --learning_rate $LEARN_RATE
