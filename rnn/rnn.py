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

class RNN(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, train=True):
        """
        Most simplest RNN which has 1 input layer, 2 hidden layers and 1 output layer.
        These 2 hidden layers are also kwown as word vector and context vector.

        Args: 
        vocab_size - The number of vocaburary
        embed_size - The size of embeddings of the input
        hidden_size - The size of the hidden layer
        train - The flag of training mode
        """
        super(RNN, self).__init__(
            l0 = L.EmbedID(vocab_size, embed_size),
            l1 = L.LSTM(embed_size, hidden_size),
            l2 = L.Linear(hidden_size, vocab_size),
        )
        self.train = train

    def reset_state(self):
        """
        We have to reset the state of all LSTMs every training batchs. Therefore, 
        when we build LSTM blocks into hidden layer, we always have to define the 
        reset_state function.
        """
        self.l1.reset_state()

    def __call__(self, cur_word):
        """
        Not forward function but __call__ function is better way in later than Chainer1.5.
        """
        x = self.l0(cur_word)
        h = self.l1(x)
        y = self.l2(h)
        return y
