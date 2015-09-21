# -*- coding: utf-8 -*-

def read_data(filename):
    data = []
    with open(filename) as fp:
        for l in fp:
            data += l.strip('\n').split(' ')
    return data

