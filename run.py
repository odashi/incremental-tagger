#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

import src.utils as utils
from src.dictionary import Dictionary
from src.postagger import POSTagger

def main():
    if len(sys.argv) != 2:
        print('usage: python postagger-runner.py model-prefix < input > output', file=sys.stderr)
        return

    model_prefix = sys.argv[1]

    # load dictionary
    word_ids = Dictionary.load(model_prefix + '.wordid')
    pos_ids = Dictionary.load(model_prefix + '.posid')

    # load and test tagger
    tagger = POSTagger.load(model_prefix)
    
    # output queue
    qs = []
    wss = []

    # input iterator
    def iterate_words():
        for l in sys.stdin:
            ls = l.strip('\n').split(' ')
            wss.append(ls)
            for w in ls:
                yield word_ids[w]

    for w, p in tagger.iterate(iterate_words()):
        qs.append(pos_ids.get_name(p))
        if len(qs) >= len(wss[0]):
            print(' '.join('%s/%s' % wq for wq in zip(wss[0], qs)))
            sys.stdout.flush()
            qs = []
            wss.pop(0)

if __name__ == '__main__':
    main()

