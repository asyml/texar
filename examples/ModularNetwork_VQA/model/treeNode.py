from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
nn = tf.keras.layers

class Bilnear(tf.keras.Model):

    def __init__(self, num_outputs, dropout=0.3):
        super(Bilnear, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.fcv = nn.Dense(self.num_features)
        self.bnv = nn.BatchNormalization(axis=-1)
        self.fcq = nn.Dense(self.num_features)
        self.fco = nn.Dense(num_outputs)
        self.dropoutw = nn.Dropout(dropout)
        self.dropoutv = nn.Dropout(dropout)

    def call(self, inputs):
        input = inputs[0]
        word = inputs[1]
        ## FC q
        word_W = self.dropoutw(word)
        w = self.fcq(word_W)
        ## FC v
        v = self.dropoutv(input)
        v = self.fcv(v)
        v = self.bnv(v)
        ## v * q_tile
        o = v * w
        o = tf.nn.relu(o)
        o = self.fco(o) 
        return tf.nn.relu(o)

class AttImgSonModule(tf.keras.Model):
    def __init__(self, num_outputs, dropout=0.3):
        super(AttImgSonModule, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.conv1 = nn.Convolution2D(self.num_features, 1, strides=1 ,padding = "same")
        self.conv2 = nn.Convolution2D(self.num_outputs, 3, strides=1 ,padding = "same")
        self.fcq_w = nn.Dense(self.num_features)
        self.fcShift1 = nn.Dense(self.num_features, activation='relu')
        self.fcShift2 = nn.Dense(self.num_features)
        self.dropoutw = nn.Dropout(dropout)
        self.dropoutv = nn.Dropout(dropout)

    def call(self, inputs):
        input = inputs[0]
        att = inputs[1]
        word = inputs[2]
        ## FC q
        word_W = self.dropoutw(word)
        weight = self.fcq_w(word_W)
        weight = tf.reshape(weight, [-1,1,1,self.num_features])
        ## FC v
        v = self.dropoutv(input)
        v = v * tf.nn.relu(1-att)
        v = self.conv1(v)
        ## attMap
        inputAttShift = tf.concat([tf.reshape(att, [-1,self.num_outputs*14*14]), word], 1)
        inputAttShift = self.fcShift1(inputAttShift)
        inputAttShift = self.fcShift2(inputAttShift)
        inputAttShift = tf.reshape(inputAttShift, [-1,1,1,self.num_features])
        ## v * q_tile
        v = v * weight * inputAttShift
        v = tf.nn.relu(v)
        # conv v
        v = self.conv2(v)
        # Normalize to single area
        return tf.nn.softmax(v, axis=-1)

class NodeBlock(tf.keras.Model):
    def __init__(self, height, num_inputs, num_outputs, input_emb_size, attNum, dropout=0.3):
        super(NodeBlock, self).__init__()
        # assert(num_inputs == num_outputs)
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = 14*14
        self.img_emb_size = num_inputs
        self.attNum = attNum
        self.height = height
        self.dropout = dropout
        self.attImgSon = AttImgSonModule(attNum, dropout)
        self.bilnear = Bilnear(num_outputs, dropout)
        self.ave = nn.AveragePooling2D(14, padding="valid")

    def call(self, inputs):
        (inputResFeat, inputAtt, inputImg, words, que_enc) = inputs
        ## cal att Map
        attMap = self.attImgSon((inputImg,inputAtt,words))
        ## x_t + son
        imFeat = tf.squeeze(self.ave(attMap*inputImg), [1,2])
        feat = tf.concat([inputResFeat, imFeat], 1)
        x_res = self.bilnear((feat,que_enc))
        outResFeat = inputResFeat+x_res
        return (outResFeat,attMap)
