#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: 小間 洋和
"""

class Vocabulary:
    def __init__(self, fname):
        self.fname = fname
        self.size = 0
        self.s2i = {}
        self.i2s = []
        if not fname is None:
            self.load_vocab()

    def stoi(self, word):
        if word in self.s2i:
            return self.s2i[word]
        return self.s2i['<unk>']

    def itos(self, id):
        if id > 0 and id < len(self.i2s):
            return self.i2s[id]
        return '<unk>'
            
    def append_words(self, words):
        for word in words:
            if word not in self.s2i:
                self.i2s.append(word)
                id = len(self.i2s) -1
                self.s2i[word] = id

    def load_vocab(self):
        self.append_words(['<unk>', '<sp>', '<s>', '</s>'])
        with open(self.fname) as f:
            for line in f:
                words = line.strip().split(' ')
                self.append_words(words)
        self.size = len(self.i2s)

    def save_vocab(self, filename):
        with open(filename, 'w') as f:
            for word in self.i2s:
                f.write(word + '\n')

    @staticmethod
    def load_from_file(filename):
        vocab = Vocabulary(None)
        with open(filename) as f:
            for word in f:
                word = word.strip()
                vocab.append_words([word])
        vocab.size = len(vocab.i2s)
        return vocab
