from __future__ import division
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import torchvision.models as models

class Bilnear(tf.keras.Model):

    def __init__(self, num_outputs, dropout=0.3):
        super(Bilnear, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.dropout = dropout

    def __call__(self, input, word):
        ## FC q
        word_W = tf.layers.dropout(word, rate=self.dropout, training = self.training)
        w = tf.layers.dense(word_W, units=self.num_features, activation=None)
        ## FC v
        v = tf.layers.dropout(input, rate=self.dropout, training = self.training)
        v = tf.layers.dense(v, units=self.num_features, activation=None)
        v = tf.layers.batch_normalization(v)
        ## v * q_tile
        o = v * w ##tf.multiply(v,w)
        o = tf.nn.relu(o)
        o = tf.layers.dense(o, units=self.num_outputs, activation=tf.nn.relu)
        return o

class AttImgSonModule(tf.keras.Model):
    def __init__(self, num_outputs, dropout=0.3):
        super(AttImgSonModule, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.dropout = dropout

    def __call__(self, input, att, word):
        ## FC q
        word_W = tf.layers.dropout(word, rate=self.dropout, training = self.training)
        weight = tf.layers.dense(word_W, units=self.num_features, activation=tf.nn.tanh)
        weight = tf.reshape(weight, [-1,self.num_features,1,1])
        ## FC v
        v = tf.layers.dropout(input, rate=self.dropout, training = self.training)
        v = v * tf.expand_dims(tf.nn.relu(1-att), 1)
        v = F.tanh(self.conv1(v))
        v = tf.layers.conv2d(v, self.num_features, kernel_size=1, padding="same", activation=tf.nn.tanh)
        ## attMap
        inputAttShift = tf.concat([tf.reshape(att, [-1,self.num_features*14*14]), word], 1)
        inputAttShift = tf.layers.dense(inputAttShift, units=self.num_features, activation=tf.nn.tanh)
        inputAttShift = tf.layers.dense(inputAttShift, units=self.num_features, activation=tf.nn.tanh)
        inputAttShift = tf.reshape(inputAttShift, [-1,self.num_features,1,1])
        ## v * q_tile
        v = v * weight * inputAttShift
        # no tanh shoulb be here
        v = tf.layers.conv2d(v, self.num_outputs, kernel_size=3, padding="same", activation=None)
        # Normalize to single area
        return tf.nn.softmax(v, axis=1)

class ResBlock(object):
    """docstring for ResBlock"""
    def __init__(self):

    def __call__(self, v1, coord_tensor):
        v = tf.concat( [v1, coord_tensor.expand(v.size(0), 2, v.size(2), v.size(3))] , 1) # (B, 130, 14, 14)
        v = tf.layers.conv2d(v, 128, kernel_size=1, padding="same", activation=tf.nn.relu)
        v = tf.layers.conv2d(v, 128, kernel_size=3, padding="same", activation=None)
        v = tf.layers.batch_normalization(v)
        v = tf.nn.relu(v)
        v = v + v1
        return v

class NodeBlock(object):

    def __init__(self, height, num_inputs, num_outputs, input_emb_size, attNum, dropout=0.3):
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = 14*14
        self.img_emb_size = num_inputs
        self.attNum = attNum
        self.height = height
        self.dropout = dropout
        self.bilnear = Bilnear(num_outputs, dropout)
        self.attImg = AttImgSonModule(attNum, dropout)

    def __call__(self, inputs):
        inputImg = sons[2]
        if self.height == 0:
            attMap = self.attImg(inputImg, words)
            imFeat = tf.matmul(tf.reshape(attMap,[-1, self.attNum, self.num_proposal]), tf.reshape(inputImg,[-1, self.img_emb_size, self.num_proposal]).transpose(1,2) )
            imFeat = tf.reshape(imFeat, [-1, self.attNum*self.img_emb_size])
            ans_feat = self.bilnear(feat,que_enc)
            return (ans_feat,attMap)
        else:
            ## son input
            inputComFeat = sons[0].sum(1).squeeze(1)
            inputAtt = sons[1].sum(1).squeeze()
            inputResFeat = sons[3].sum(1).squeeze(1)
            if inputAtt.dim() == 2: inputAtt = inputAtt.unsqueeze(0)
            ## cal att Map
            attMap = self.attImgSon(inputImg,inputAtt,words)
            ## x_t + son
            imFeat = tf.matmul(tf.reshape(attMap,[-1, self.attNum, self.num_proposal]), tf.reshape(inputImg,[-1, self.img_emb_size, self.num_proposal]).transpose(1,2) )
            imFeat = tf.reshape(imFeat, [-1, self.attNum*self.img_emb_size])
            feat = tf.concat([inputResFeat, imFeat], 1)#feat = inputComFeat + imFeat#
            x_res = self.bilnear(feat,que_enc)
            outResFeat = inputResFeat+x_res
            return (outResFeat,attMap)

class CNN(nn.Module):
    """docstring for tree"""
    def __init__(self):
        super(CNN, self).__init__()
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = self.featMapH*self.featMapW
        # nets
        self.res1 = ResBlock()
        self.res2 = ResBlock()

        def cvt_coord(i):
            return [(i/self.featMapW-(self.featMapH//2))/(self.featMapH/2.), (i%self.featMapW-(self.featMapW//2))/(self.featMapW/2.)]
        np_coord_tensor = np.zeros((self.num_proposal, 2))
        for i in range(self.num_proposal):
            np_coord_tensor[i,:] = np.array( cvt_coord(i) )
        self.coord_tensor = tf.convert_to_tensor(np_coord_tensor)
        self.coord_tensor = tf.reshape(1,self.featMapH,self.featMapW,2)
        # self.coord_tensor = self.coord_tensor.view(1,self.featMapH,self.featMapW,2).transpose_(2,3).transpose_(1,2)

    def __call__(self, img):
        img = tf.concat([img, tf.tile(self.coord_tensor, [img[0], 1, 1, 1])], 1)
        img = tf.layers.conv2d(img, 128, kernel_size=3, padding="same", activation=None)
        img = tf.layers.batch_normalization(img)
        img = tf.nn.relu(img)

        img = self.res1(img, self.coord_tensor)
        img = self.res2(img, self.coord_tensor)

        return img

class tree_attention_abstract_DP(nn.Module):
    """docstring for tree"""
    def __init__(self, opt):
        super(tree_attention_abstract_DP, self).__init__()
        self.featMapH = 14
        self.featMapW = 14
        self.batch_size = opt.batch_size
        self.num_proposal = self.featMapH * self.featMapW
        self.input_emb_size = opt.sentence_emb
        self.dropout = opt.dropout
        self.img_emb_size = opt.img_emb
        self.gpu = opt.gpu
        self.common_embedding_size = opt.commom_emb
        self.enc_mode = opt.encode
        self.sent_len = opt.sent_len
        # self.num_output = opt.num_output # mc number 

        else: self.lookup = nn.Embedding(opt.vocab_size + 1, opt.word_emb_size, padding_idx = 0) #mask zero
        self.q_LSTM = nn.LSTM(300,1024,1,bidirectional=True)#droput not exist for 1-layer RNN
        self.CNN = CNN()

    def __call__(self, que, img, tree):
        self.batch_size = img.shape[0]
        self.bmax_height = max((t[-1]['height'] for t in tree)) + 1
        #--------get selected map---------------
        # return a (B, H ,[x,x,x]) list
        def get_selected_map(tree, maxh):
            selected_map = []
            que_len = []
            for i in range(self.batch_size):
                length = 0
                tmap = [[] for _ in range(maxh)]
                for j in range(len(tree[i])):
                    h = tree[i][j]['height']
                    tmap[h].append(j)
                    length+=len(tree[i][j]['word'])
                selected_map.append(tmap)
                que_len.append(length)
            return selected_map, que_len
        #--------------end----------------------
        node_values = [[len(tree[i]) * [o] for i in range(self.batch_size)] for o in self.init_node_values()] # init node_values
        selected_map, que_len = get_selected_map(tree, self.bmax_height)
        #-------img prepro------------------
        que_enc, que_enc_sent = self._encoder(que) # (batch_size, emb)
        img = tf.nn.l2_normalize(img, axis=2)
        img = self.CNN(img)
        #---------end-----------------------

        def get_tw_tn(bs_sum, height):
            tmp_word_inputs = tf.zeros(bs_sum,self.sent_len)
            tmp_q_inputs = tf.zeros(bs_sum,self.input_emb_size)
            tree_nodes = []
            cnt = 0
            for i in range(self.batch_size):
                for j in range(len(selected_map[i][height])): 
                    t_node = tree[i][selected_map[i][height][j]]
                    word_idx = self.sent_len - que_len[i] + t_node['index']-1
                    assert word_idx>=0 and word_idx<self.sent_len, 'word_idx: {}, {}, {}, {}'.format(word_idx,self.sent_len,que_len[i],t_node['index'])
                    tmp_q_inputs[cnt] = que_enc_sent[word_idx][i]
                    tree_nodes.append(t_node)
                    cnt += 1
            return tmp_word_inputs, tree_nodes, ijlist, tmp_q_inputs

        def put_back_nv(tmp_outputs, node_values, height):
            for k in range(len(tmp_outputs)):
                cnt = 0
                for i in range(len(node_values[k])):
                    for j in selected_map[i][height]:
                        node_values[k][i][j] = tmp_outputs[k][cnt]
                        cnt += 1

        for height in range(self.bmax_height):
            bs_sum = sum([len(selected_map[i][height]) for i in range(self.batch_size)])
            ijlist = []
            for i in range(self.batch_size): 
                for j in selected_map[i][height]: ijlist.append((i, j))
            tmp_inputs, node_que_enc = self.load_tmp_values(img, que_enc, tree, ijlist, node_values, height) # [4, TS(batch_sum_mnum, maxson, f1, f2...) ]
            tmp_word_inputs, tree_nodes, ijlist, tmp_q_inputs = get_tw_tn(bs_sum, height)
            tmp_outputs = self.node[height](tmp_inputs, tmp_q_inputs, tree_nodes, node_que_enc)
            put_back_nv(tmp_outputs, node_values, height)

        predict = self.root_to_att(que_enc, img, tree, node_values)
        return predict, node_values

    def _encoder(self, sentence, enc_mode=None):
        # sentence vocab begins from 1.And 0 means no word
        # input size: -1 * sentence_length
        # output size: -1 * embedding_size
        emb = self.lookup(sentence.view(-1, self.sent_len))
        qenc, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.contrib.rnn.GRUCell(1024),
            tf.contrib.rnn.GRUCell(1024),
            emb,
            sequence_length=None,
            time_major=False)
        qenc = tf.concat(qenc, 2)
        qenc = tf.nn.l2_normalize(qenc, axis=-1)
        enc = qenc[-1]
        return enc, qenc

    def init_node_values(self):
        raise NotImplementedError

    def load_tmp_values(self, img, tree, ijlist, node_values, height):
        raise NotImplementedError

    def root_to_att(self, img, node_values):
        raise NotImplementedError