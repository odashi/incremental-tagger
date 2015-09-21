# -*- coding: utf-8 -*-

import sys
import shutil

from .perceptron import *

class POSTagger:
    def __init__(self, word_ngram_size, pos_ngram_size, word_window_size, pos_history_size):
        if word_window_size < 0:
            raise RuntimeError('invalid word_window_size')
        if pos_history_size < 0:
            raise RuntimeError('invalid pos_history_size')
        if word_ngram_size > 2 * word_window_size + 1:
            raise RuntimeError('too large word_ngram_size')
        if pos_ngram_size > pos_history_size:
            raise RuntimeError('too large pos_ngram_size')

        self.__word_ngram_size = word_ngram_size
        self.__pos_ngram_size = pos_ngram_size
        self.__word_window_size = word_window_size
        self.__pos_history_size = pos_history_size

        # dummy members
        self.__classifier = None

    @staticmethod
    def __get_entry(list, i):
        l = len(list)
        return list[(i + l) % l]

    def __make_feature_list(self, word_list, position, pos_history):
        feature_list = []
        ws = self.__word_window_size
        ps = self.__pos_history_size

        # word n-gram
        names = []
        for i in range(-ws, ws + 1):
            names.append('W%+d=%d' % (i, self.__get_entry(word_list, position + i)))
        for n in range(self.__word_ngram_size):
            for i in range(len(names) - n):
                feature_list.append(','.join(names[i : i + n + 1]))
        
        # POS n-gram
        names = []
        for i in range(ps):
            names.append('P%+d=%d' % (i - ps, pos_history[i]))
        for n in range(self.__pos_ngram_size):
            for i in range(len(names) - n):
                feature_list.append(','.join(names[i : i + n + 1]))
        
        #print(' '.join(feature_list))
        return feature_list

    def train(self, pos_vocab_size, train_word_list, train_pos_list, dev_word_list, dev_pos_list, max_iteration, model_prefix):
        train_size = len(train_word_list)
        dev_size = len(dev_word_list)
        if (len(train_pos_list) != train_size):
            raise RuntimeError('number of train-set words and POSs are different : %d != %d' % (train_size, len(train_pos_list)))
        if (len(dev_pos_list) != dev_size):
            raise RuntimeError('number of dev-set words and POSs are different : %d != %d' % (dev_size, len(dev_pos_list)))

        self.save_settings(model_prefix)

        self.__classifier = AveragedPerceptronTrainer(pos_vocab_size)
        #self.__classifier = L1RegularizedPerceptronTrainer(pos_vocab_size)
        best_iteration = -1
        best_acc = -1

        for iteration in range(max_iteration):
            print('iteration %d:' % (iteration + 1), file=sys.stderr)

            # train
            if self.__pos_history_size > 0:
                pos_history = train_pos_list[-self.__pos_history_size:] # dummy POS
            else:
                pos_history = []

            for k in range(train_size):
                feature_list = self.__make_feature_list(train_word_list, k, pos_history)
                best_pos = self.__classifier.train(train_pos_list[k], feature_list)
                pos_history.append(best_pos)
                pos_history.pop(0)

                if (k + 1) % 100 == 0:
                    print('\r  trained: %d/%d' % (k + 1, train_size), end='', file=sys.stderr)
            
            print('\r  trained: %d/%d' % (train_size, train_size), file=sys.stderr)

            # dev test
            acc = self.test(dev_word_list, dev_pos_list)
            if acc > best_acc:
                best_iteration = iteration
                best_acc = acc
                print('  best!', file=sys.stderr)

            # save temporary data
            self.save_model(model_prefix + '.%03d' % (iteration + 1))

        print('done!', file=sys.stderr)
        print('copying best model (iteration=%d, acc=%.2f%%) ...' % (best_iteration + 1, 100.0 * best_acc), file=sys.stderr)
        shutil.copyfile(model_prefix + '.%03d' % (best_iteration + 1) + '.model', model_prefix + '.model')

    def test(self, test_word_list, test_pos_list):
        test_size = len(test_word_list)
        if (len(test_pos_list) != test_size):
            raise RuntimeError('number of test-set words and POSs are different')
        
        if self.__pos_history_size > 0:
            pos_history = test_pos_list[-self.__pos_history_size:] # dummy POS
        else:
            pos_history = []
        correct = 0

        for k in range(test_size):
            feature_list = self.__make_feature_list(test_word_list, k, pos_history)
            best_pos = self.__classifier.predict(feature_list)
            pos_history.append(best_pos)
            pos_history.pop(0)

            if best_pos == test_pos_list[k]:
                correct += 1
            if (k + 1) % 100 == 0:
                print('\r  tested: %d/%d' % (k + 1, test_size), end='', file=sys.stderr)

        print('\r  tested: %d/%d' % (test_size, test_size), file=sys.stderr)
            
        acc = correct / test_size
        print('  accuracy: %.2f%% (%d/%d)' % (100.0 * acc, correct, test_size), file=sys.stderr)
        return acc

    def get_appended_word_iterator(self, wordid_iterator):
        for _ in range(self.__word_window_size):
            yield -1
        for w in wordid_iterator:
            yield w
        for _ in range(self.__word_window_size):
            yield -1

    def iterate(self, wordid_iterator):
        if self.__pos_history_size > 0:
            pos_history = [-1] * self.__pos_history_size
        else:
            pos_history = []

        window_length = 2 * self.__word_window_size + 1
        word_window = []

        for w in self.get_appended_word_iterator(wordid_iterator):
            word_window.append(w)
            if len(word_window) < window_length:
                continue

            # estimation
            feature_list = self.__make_feature_list(word_window, self.__word_window_size, pos_history)
            best_pos = self.__classifier.predict(feature_list)
            pos_history.append(best_pos)
            pos_history.pop(0)
            yield word_window[self.__word_window_size], best_pos
            word_window.pop(0)
        
    def save_settings(self, model_prefix):
        print('saving POS tagger settings ...', file=sys.stderr)
        with open(model_prefix + '.settings', 'w') as fp:
            fp.write('WORD_NGRAM\t%d\n' % self.__word_ngram_size)
            fp.write('POS_NGRAM\t%d\n' % self.__pos_ngram_size)
            fp.write('WORD_WINDOW\t%d\n' % self.__word_window_size)
            fp.write('POS_HISTORY\t%d\n' % self.__pos_history_size)

    def save_model(self, model_prefix):
        print('saving POS tagger model ...', file=sys.stderr)
        self.__classifier.save(model_prefix + '.model')
    
    @staticmethod
    def load(model_prefix):
        print('loading POS tagger settings/model ...', file=sys.stderr)

        # load settings
        sett = {}
        with open(model_prefix + '.settings') as fp:
            for l in fp:
                ls = l.split()
                sett[ls[0]] = int(ls[1])

        ret = POSTagger(sett['WORD_NGRAM'], sett['POS_NGRAM'], sett['WORD_WINDOW'], sett['POS_HISTORY'])
        ret.__classifier = Perceptron.load(model_prefix + '.model')
        
        return ret

