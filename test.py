#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

import src.utils as utils
from src.dictionary import Dictionary
from src.postagger import POSTagger

def main():
    if len(sys.argv) != 3:
        print('usage: python postagger-test.py', file=sys.stderr)
        print('                <str: test prefix>', file=sys.stderr)
        print('                <str: model prefix>', file=sys.stderr)
        return

    test_prefix = sys.argv[1]
    model_prefix = sys.argv[2]

    print('loading data ...', file=sys.stderr)

    # load test data
    test_words = [w.lower() for w in utils.read_data(test_prefix + '.words')]
    test_pos = utils.read_data(test_prefix + '.pos')

    # load dictionary
    word_ids = Dictionary.load(model_prefix + '.wordid')
    pos_ids = Dictionary.load(model_prefix + '.posid')

    # make word/POS IDs
    test_wids = [word_ids[w] for w in test_words]
    test_pids = [pos_ids[w] for w in test_pos]

    # load and test tagger
    tagger = POSTagger.load(model_prefix)
    tagger.test(test_wids, test_pids)

if __name__ == '__main__':
    main()

