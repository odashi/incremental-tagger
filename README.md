Incremental Part-of-speech Tagger
=================================

A part-of-speech tagger using averaged perceptrong and tagging history


Usage
-----

You first should prepare 4 files:
* Preliminary segmented word corpus (train.words, dev.words)
* POS corpus corresponding above words (train.pos, dev.pos)

To train the tagging model using:
* Word unigram
* POS bigram
* Word window with width = 3
* 5 POS histories,
* 10 training epoch
run below:

    postagger-train.py train dev model 1 2 3 5 10


Contributors
------------

* Yusuke Oda (@odashi)

We are counting more contributions from you.


Contact
-------

If you find an issue, please contact Y.Oda
* yus.takara (at) gmail.com
* @odashi_t on Twitter

