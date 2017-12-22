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
    source_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'data/translation/de-en/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 32 # alias = N
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    num_epochs = 20




