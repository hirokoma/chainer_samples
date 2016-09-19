#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: komah
"""

import argparse
import sys, time

import numpy as np
import chainer
from chainer import cuda, gradient_check, optimizers, serializers, utils
from chainer import Function, Variable, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class RCNN(Chain):
    def __init__(self, dim_word, dim_context, dim_sentence, class_size, train=True):
        super(RNN, self).__init__(
            l1_fwd  = L.LSTM(dim_word, dim_context),
            l1_bwd = L.LSTM(dim_word, dim_context),
            l2 = L.Linear(dim_word + 2*dim_context , dim_sentence),
            l3 = L.Linear(dim_sentence, class_size)
        )
        self.train = train

    def __call__(self, x_list):

        # forward step
        self.l1_fwd.reset_state()
        cl_list = []
        for x in x_list:
            c = self.l1_fwd(x)
            cl_list.append(c)

        # backward step
        self.l1_bwd.reset_state()
        cr_list = []
        for x in reversed(x_list):
            c = self.l1_fwd(x)
            cr_list.insert(0, c)

        # concat left-context vec, word vec and right-context vec
        y1_list = []
        for cl, x, cr in zip(cl_list, x_list, cr_list):
            y1_list.append(F.concat((cl,x,cr)))

        # make latent semantic vec
        y2_list = []
        for c in c_list:
            y2 = F.tanh( self.l2(c) )
            y2_list.append(y2)

        # max pooling
        y3 = y2_list[0]
        for y2 in y2_list[1:]:
            y3 = F.maximum(y2, y3)

        # full-connected
        y = self.l3(y3)

        return y
