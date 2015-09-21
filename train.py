#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

import src.utils as utils
from src.dictionary import Dictionary
from src.postagger import POSTagger

def main():
    if len(sys.argv) != 9:
        print('usage: python postagger-train.py', file=sys.stderr)
        print('                <str: train prefix>', file=sys.stderr)
        print('                <str: dev prefix>', file=sys.stderr)
        print('                <str: model prefix>', file=sys.stderr)
        print('                <int: word n-gram size>', file=sys.stderr)
        print('                <int: POS n-gram size>', file=sys.stderr)
        print('                <int: word window size>', file=sys.stderr)
        print('                <int: POS history size>', file=sys.stderr)
        print('                <int: max iteration>', file=sys.stderr)
        return

    train_prefix = sys.argv[1]
    dev_prefix = sys.argv[2]
    model_prefix = sys.argv[3]
    word_ngram_size = int(sys.argv[4])
    pos_ngram_size = int(sys.argv[5])
    word_window_size = int(sys.argv[6])
    pos_history_size = int(sys.argv[7])
    max_iteration = int(sys.argv[8])

    print('loading data ...', file=sys.stderr)

    # load train/dev data
    train_words = [w.lower() for w in utils.read_data(train_prefix + '.words')]
    train_pos = utils.read_data(train_prefix + '.pos')
    dev_words = [w.lower() for w in utils.read_data(dev_prefix + '.words')]
    dev_pos = utils.read_data(dev_prefix + '.pos')

    # make dictionary
    word_ids = Dictionary(train_words, frozen=True)
    pos_ids = Dictionary(train_pos, frozen=True)
    word_ids.save(model_prefix + '.wordid')
    pos_ids.save(model_prefix + '.posid')

    # make word/POS IDs
    train_wids = [word_ids[w] for w in train_words]
    train_pids = [pos_ids[w] for w in train_pos]
    dev_wids = [word_ids[w] for w in dev_words]
    dev_pids = [pos_ids[w] for w in dev_pos]

    # train
    tagger = POSTagger(word_ngram_size, pos_ngram_size, word_window_size, pos_history_size)
    tagger.train(len(pos_ids), train_wids, train_pids, dev_wids, dev_pids, max_iteration, model_prefix)

if __name__ == '__main__':
    main()

