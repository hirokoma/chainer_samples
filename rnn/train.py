#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: komah
"""
import argparse
import sys, time
import os.path

import numpy as np
import chainer
from chainer import cuda, gradient_check, optimizers, serializers, utils
from chainer import Function, Variable, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from rnn import RNN
from vocabulary import Vocabulary

def argument():
    parser = argparse.ArgumentParser()
    # For training
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_text', default='')
    parser.add_argument('--n_epoch', default=10, type=int)
    parser.add_argument('--n_batch', default=256, type=int)
    # parser.add_argument('--unchain', action='store_true', default=False)
    # parser.add_argument('--dropout', action='store_true', default=False)
    # parser.add_argument('--gradgrip', action='store_true', default=False)
    
    # For evaluation
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--evaluate-text', default='')

    # For generation
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--generate_text', default='', help ='seed text of generation')

    # Network
    parser.add_argument('--n_embed', default=200, type=int)
    parser.add_argument('--n_hidden', default=200, type=int)

    # Vocaburary
    parser.add_argument('--vocab_file', default='')

    # Model
    parser.add_argument('--model', default='')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()
    return args

def file_exists(filename):
    return os.path.isfile(filename)

def train(args):

    if not file_exists(args.train_text):
        return

    """
    Setup vocabulary
    """
    if file_exists(args.vocab_file):
        vocab = Vocabulary.load_from_file(args.vocab_file)
    elif file_exists(args.train_text):
        vocab = Vocabulary(args.train_text)
        vocab.save_vocab(args.vocab_file)
    print('Vocaburlary loaded.')
    print('  * vocab size: {}'.format(vocab.size))

    """
    Setup model
    """
    n_vocab = vocab.size
    n_embed = args.n_embed
    n_hidden = args.n_hidden

    rnn = RNN(n_vocab, n_embed, n_hidden)
    rnn.reset_state()
    model = L.Classifier(rnn)
    model.compute_accuracy = False
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    print('RNN model loaded.')
    print('  * n_vocab: {}'.format(n_vocab) )
    print('  * n_embed: {}'.format(n_embed) )
    print('  * n_hidden: {}'.format(n_hidden) )

    # Setup GPU
    if args.use_gpu:
        xp = cuda.cupy
        cuda.get_device(0).use()
        model.to_gpu()
    else:
        xp = np

    """
    Setup data
    """
    sequences = []
    for line in open(args.train_text, 'r'):
        line = line.strip()
        # ignore the sentence which length is lower than 10
        if len(line) < 10:
            continue
        # sequence of integers in 1 sentence
        sequence = []

        # make sequence
        sequence.append( vocab.stoi('<s>') )
        [sequence.append( vocab.stoi(word) ) for word in line.split(' ')]
        sequence.append( vocab.stoi('</s>') )

        # append to sequences list
        sequences.append(sequence)

    length_of_longest_sentence = max([len(sequence) for sequence in sequences])
    n_data = len(sequences)
    print('Data loaded.')
    print('  * length of sentence: {}'.format(length_of_longest_sentence))
    print('  * number of training data: {}'.format(n_data))

    # fill <sp> into empty elements
    for i, sequence in enumerate(sequences):
        length = len(sequence)
        while length < length_of_longest_sentence:
            sequences[i].append( vocab.stoi('<sp>') )
            length = len(sequences[i])

    """
    Training loop
    """
    n_epoch = args.n_epoch
    n_batch = args.n_batch
    
    print('Training started.')
    print('  * number of epochs: {}'.format(n_epoch))
    print('  * batch size: {}'.format(n_batch))
    for epoch in range(1, n_epoch+1):
        for jump in range(0, n_data - n_batch , n_batch):

            x_list = Variable(xp.array(
                sequences[jump : jump + n_batch],
            dtype=xp.int32).transpose())
            
            loss = 0
            count = 0
            rnn.reset_state()
            seqlen = len(x_list[1:])
            for cur_word, next_word in zip(x_list, x_list[1:]):
                loss += model(cur_word, next_word)
                count += 1
                if count % 30 == 0 or count == seqlen:
                    model.cleargrads()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()
            if jump%10 == 0:
                print('  ** epoch={}, batch idx={}, loss={}, progress={}'.format(
                    epoch,
                    jump,
                    xp.mean(loss.data),
                    str(float(epoch * (n_data - n_batch) / n_batch + jump ) / (n_epoch * (n_data - n_batch) / n_batch))[:5]
                ))
        if epoch % 1 == 0:
            outfile = args.model + '-{}epoch.model'.format(epoch)
            serializers.save_npz(outfile, model)
            outfile = args.model + '-{}epoch.optimizer'.format(epoch)
            serializers.save_npz(outfile, optimizer)

def main():
    args = argument()
    if args.train:
        train(args)
    #else:
        #eval(args)

if __name__ == '__main__':
    main()
