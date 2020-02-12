#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created max_entropy by rjw at 19-3-11 in WHU.
"""

import numpy as np
from maxentropy.skmaxent import MinDivergenceModel


# the maximised distribution must satisfy the mean for each sample
def get_features():
    def f0(x):
        return x

    return [f0]

def get_max_entropy_distribution(mean):
    SAMPLESPACE = np.arange(10)
    features = get_features()

    model = MinDivergenceModel(features, samplespace=SAMPLESPACE, algorithm='CG')

    # set the desired feature expectations and fit the model
    X = np.array([[mean]])
    model.fit(X)

    return model.probdist()