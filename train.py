#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: 小間 洋和
"""
import argparse
import sys, time

import numpy as np
import chainer
from chainer import cuda, gradient_check, optimizers, serializers, units
from chainer import Function, Variable, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from rnn import RNN
from vocaburary import Vocaburary

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('file')
    parser.add_argument('--n_embed', default=200, type=int)
    parser.add_argument('--n_vocab', default=3000, type=int)
    parser.add_argument('--n_hidden', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default="model")
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--unchain', action='store_true', default=False)
    args = parser.parse_args()
    return args

def train(args):
    """
    Train RNNLM
    """
    vocab = Vocabulary(args.file)
    vocab_size = 


    rnn = RNN(args.n_vocab, args.n_embed, args.n_hidden)
    model = L.Classifier(rnn)
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    data = 

    if args.use_gpu:
        xp = cuda.cupy
        cuda.get_device(0).use()
        model.to_gpu()
    else:
        xp = np

    """
    compute_loss (時間方向の順伝播)
    """
    loss = 0
    for current_word, next_word in zip()

def main():
    args = argument()
    if args.mode == 'train':
        train(args)
    else:
        eval(args)

if __name__ == '__main__':
    main()
