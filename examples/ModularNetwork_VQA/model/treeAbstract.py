from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
nn = tf.keras.layers

class ResBlock(tf.keras.Model):
    """docstring for ResBlock"""
    def __init__(self):
        super(ResBlock, self).__init__()
        # nets
        self.conv1 = nn.Convolution2D(128, 1, strides=1 ,padding = "same")
        self.conv2 = nn.Convolution2D(128, 3, strides=1 ,padding = "same")
        self.bn = nn.BatchNormalization(axis=-1)

    def call(self, v):
        v1 = tf.nn.relu(self.conv1(v)) # (B, 128, 14, 14)

        v = self.conv2(v1)
        v = self.bn(v)
        v = tf.nn.relu(v)

        v = v + v1
        return v

class CNN(tf.keras.Model):
    """docstring for tree"""
    def __init__(self):
        super(CNN, self).__init__()
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = self.featMapH*self.featMapW
        # nets
        self.conv = nn.Convolution2D(128, 3, strides=1 ,padding = "same")
        self.bn = nn.BatchNormalization(axis=-1)
        self.res1 = ResBlock()
        self.res2 = ResBlock()

    def call(self, img):
        img = self.bn(self.conv(img))
        img = tf.nn.relu(img)

        img = self.res1(img)
        img = self.res2(img)

        return img

class tree_attention_abstract_DP(tf.keras.Model):
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
        self.sent_len = opt.sent_len

        self.lookup = nn.Embedding(opt.vocab_size + 1, opt.word_emb_size, mask_zero = 0, input_length=opt.sent_len) #mask zero
        self.q_LSTM = nn.Bidirectional(nn.LSTM(1024, return_sequences=True), merge_mode='concat')
        self.CNN = CNN()
        self.ave = nn.AveragePooling2D(14, padding="valid")

    def call(self, inputs):
        que = inputs[0]
        img = inputs[1]
        tree = self.trees
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
        img = tf.nn.l2_normalize(img, axis=1)
        img = self.CNN(img)
        #---------end-----------------------

        def get_tw_tn(bs_sum, height):
            tree_nodes = []
            batch_emb = []
            for i in range(self.batch_size):
                for j in range(len(selected_map[i][height])): 
                    t_node = tree[i][selected_map[i][height][j]]
                    word_idx = self.sent_len - que_len[i] + t_node['index']-1
                    assert word_idx>=0 and word_idx<self.sent_len, 'word_idx: {}, {}, {}, {}'.format(word_idx,self.sent_len,que_len[i],t_node['index'])
                    batch_emb.append(que_enc_sent[i][word_idx])
                    tree_nodes.append(t_node)
            tmp_q_inputs = tf.stack(batch_emb)
            return tree_nodes, tmp_q_inputs

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
            tree_nodes, tmp_q_inputs = get_tw_tn(bs_sum, height)
            tmp_outputs = self.node[height](tmp_inputs + (tmp_q_inputs, node_que_enc))
            put_back_nv(tmp_outputs, node_values, height)

        predict = self.root_to_att(que_enc, img, tree, node_values)
        return predict, node_values

    def _encoder(self, sentence):
        # sentence vocab begins from 1.And 0 means no word
        # input size: -1 * sentence_length
        # output size: -1 * embedding_size
        emb = self.lookup(sentence)
        qenc = self.q_LSTM(emb)
        qenc = tf.nn.l2_normalize(qenc, axis=-1)
        enc = qenc[:,-1,:]
        return enc, qenc

    def structured_input_fn(self, trees):
        self.trees = trees

    def init_node_values(self):
        raise NotImplementedError

    def load_tmp_values(self, img, tree, ijlist, node_values, height):
        raise NotImplementedError

    def root_to_att(self, img, node_values):
        raise NotImplementedError