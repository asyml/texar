# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'data/translation/de-en/train.tags.de-en.de'
    target_train = 'data/translation/de-en/train.tags.de-en.en'
    #source_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    #target_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    source_test = '/tmp/t2t_datagen/newstest2014.tok.bpe.32000.en'
    target_test = '/tmp/t2t_datagen/newstest2014.tok.bpe.32000.de'
    # training
    batch_size = 32 # this is used only for test

    maxlen = 256 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 0 # words whose occurred less than min_cnt are encoded as <UNK>.



