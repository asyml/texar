import json
import os
import tensorflow as tf
from .treeAbstract import tree_attention_abstract_DP
from . import treeNode
nn = tf.keras.layers

class tree_attention_Residual(tree_attention_abstract_DP):
	def __init__(self, opt):
		super(tree_attention_Residual, self).__init__(opt)
		self.num_node_feature = 256
		self.max_height = 12
		self.dropout=opt.dropout
		self.attNum = 1
		self.out_vocab_size = opt.out_vocab_size
		##########################################################
		self.node = []
		for i in range(self.max_height):
			self.node.append(treeNode.NodeBlock(i, self.img_emb_size, self.num_node_feature, self.input_emb_size, self.attNum, self.dropout))
		##########################################################
		## AnsModule
		self.fc0 = nn.Dense(512)
		self.bn0 = nn.BatchNormalization(axis=-1)
		self.fc1 = nn.Dense(1024)
		self.bn1 = nn.BatchNormalization(axis=-1)
		self.fc2 = nn.Dense(opt.out_vocab_size)

	def init_node_values(self):
		return (None, None)

	def load_tmp_values(self, img, que_enc, tree, tree_list, node_values, height):
		batch_i = []
		if height == 0:
			input0 = tf.zeros([len(tree_list), self.num_node_feature])
			input1 = tf.zeros([len(tree_list), self.featMapH, self.featMapW, 1])
			for i, v in enumerate(tree_list):
				batch_i.append(v[0])
			input2 = tf.gather(img, batch_i)
			node_que_enc = tf.gather(que_enc, batch_i)
			return (input0, input1, input2), node_que_enc
		else:
			batch_h = []
			batch_att = []
			for i, v in enumerate(tree_list):
				batch_i.append(v[0])
				ibatch = v[0]
				jpos = v[1]
				child_h = []
				child_att = []
				for child_cnt in range(len(tree[ibatch][jpos]['child'])):
					child_pos = tree[ibatch][jpos]['child'][child_cnt]
					child_h.append(node_values[0][ibatch][child_pos])
					child_att.append(node_values[1][ibatch][child_pos])
				batch_h.append(tf.add_n(child_h))
				batch_att.append(tf.add_n(child_att))
			input0 = tf.stack(batch_h)
			input1 = tf.stack(batch_att)
			input2 = tf.gather(img, batch_i)
			node_que_enc = tf.gather(que_enc, batch_i)
		return (input0,input1,input2), node_que_enc

	def answer(self, feat):
		x_ = self.fc0(feat)
		x_ = self.bn0(x_)
		x_ = tf.nn.relu(x_)
		x_ = self.fc1(x_)
		x_ = self.bn1(x_)
		x_ = tf.nn.relu(x_)
		x_ = self.fc2(x_)
		return x_

	def root_to_att(self, que_enc, img, tree, node_values):
		res = []
		for i in range(self.batch_size):
			j = len(tree[i]) - 1
			res.append(node_values[0][i][j])

		predict = self.answer(tf.stack(res))
		return predict