# -*- coding: utf-8 -*-

class Dictionary:
    def __init__(self, seeds = None, frozen = False):
        self.__table = {}
        self.__reverse = {}
        self.__frozen = False # dummy
        if seeds:
            for w in seeds:
                self[w] # register an entry
        if frozen:
            self.freeze()

    def __iter__(self):
        return iter(self.__table)

    def __len__(self):
        return len(self.__table)

    def __contains__(self, x):
        return x in self.__table

    def __getitem__(self, key):
        try:
            return self.__table[key]
        except:
            if self.__frozen:
                return -1
            else:
                newval = len(self.__table)
                self.__table[key] = newval
                self.__reverse[newval] = key
                return newval

    def get_name(self, id):
        try:
            return self.__reverse[id]
        except:
            return None

    def freeze(self):
        self.__frozen = True

    def save(self, filename):
        with open(filename, 'w') as fp:
            for key, value in self.__table.items():
                fp.write('%s\t%s\t%d\n' % (key.__class__.__name__, key, value))

    @staticmethod
    def load(filename):
        ret = Dictionary()
        with open(filename) as fp:
            for l in fp:
                ls = l.strip('\n').split('\t')
                cls = __builtins__[ls[0]]
                key = cls(ls[1])
                val = int(ls[2])
                ret.__table[key] = val
                ret.__reverse[val] = key
        ret.freeze()
        return ret

