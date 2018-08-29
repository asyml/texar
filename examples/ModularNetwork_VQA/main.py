from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import shutil
from tqdm import tqdm
import json
import model.tree as M
from data.dataset import clevrDataset

tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

opt = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32, """training batch size""")
tf.app.flags.DEFINE_integer('max_epoch', 1000, """max epoches""")
tf.app.flags.DEFINE_float('lr', 9e-5, 'learning Rate. Default=0.0002')
tf.app.flags.DEFINE_float('weight_decay', 0, 'weight decay. Default=0.0002')
tf.app.flags.DEFINE_float('beta1', 0.9, 'adam Beta1. Default=0.5')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam Beta2. Default=0.999')
tf.app.flags.DEFINE_integer('threads', 0, 'number of threads for data loader to use')
tf.app.flags.DEFINE_integer('seed', 1, 'random seed to use. Default=123')
tf.app.flags.DEFINE_integer('display_interval', 100, 'display loss after each several iters')
tf.app.flags.DEFINE_bool('display_att', False, 'whether display attention map or not')
tf.app.flags.DEFINE_bool('gpu', True, 'use gpu?')
# model settings
tf.app.flags.DEFINE_integer('sentence_emb', 2048, 'embedding size of sentences')
tf.app.flags.DEFINE_integer('word_emb_size', 300, 'embedding size of words')
tf.app.flags.DEFINE_integer('img_emb', 128, 'embedding size of images')
tf.app.flags.DEFINE_integer('pre_emb', 310, 'embedding size of preprocess')
tf.app.flags.DEFINE_integer('commom_emb', 256, 'commom embedding size of sentence and image')
tf.app.flags.DEFINE_integer('vocab_size', 81, 'word vocabulary size')
tf.app.flags.DEFINE_integer('out_vocab_size', 29, 'answer vocabulary size')
tf.app.flags.DEFINE_integer('sent_len', 45, 'maximum length of a question')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout ration. Default=0.0')

# file settings
tf.app.flags.DEFINE_string('vocab_dir', './clevr', 'dir of qa pairs')
tf.app.flags.DEFINE_string('tree_dir', './clevr/parsed_tree', 'dir of qa pairs')
tf.app.flags.DEFINE_string('clevr_img_h5', './clevr/clevr_res101', 'dir of vqa image feature')
tf.app.flags.DEFINE_string('resume', None, 'resume file name')
tf.app.flags.DEFINE_string('logdir', 'logs/clevr', 'dir to tensorboard logs')

########################################################
## 			         Load Data	        		   	  ##
########################################################
dataloader_train = clevrDataset(opt, 'train')
dataloader_test =  clevrDataset(opt, 'val')
print('dataset loaded')
########################################################
## 			         Basic Settings			   	  	  ##
########################################################
if not os.path.isdir(opt.logdir): os.mkdir(opt.logdir)
if opt.logdir is not None:
	f = open(opt.logdir + '/params.txt','w')
	print(opt, file = f)
	f.close()
	shutil.copyfile('main.py', opt.logdir + '/main.py')
	if os.path.isdir(opt.logdir + '/model'): shutil.rmtree(opt.logdir + '/model')
	shutil.copytree('model/', opt.logdir + '/model/')
if opt.display_att == True:
	mvizer = vizer(opt, writer)
device = "/gpu:0"
########################################################
## 			       Build Model 						  ##
########################################################
print('Building network...')

net = M.tree_attention_Residual(opt)

optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)

finished_epoch = 0

root = tfe.Checkpoint(optimizer=optimizer,
                      model=net,
                      optimizer_step=tf.train.get_or_create_global_step())

if opt.resume is not None:
	print('resume from ' + opt.resume)
	root.restore(tf.train.latest_checkpoint(opt.logdir))
print(net)

########################################################
## 					  	Loss				  		  ##
########################################################
def loss(logits, labels):
	return tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits, labels=labels))
########################################################
## 					  Train Model			  		  ##
########################################################

def train(epoch):
	print('Trainning...')

	global_step=tf.train.get_or_create_global_step()

	dataloader_train.reset_state()

	for _ in tqdm(range(dataloader_train.num_iter)):
		with tf.contrib.summary.record_summaries_every_n_global_steps(opt.display_interval, global_step=tf.train.get_or_create_global_step()):
			question, image, trees, labels, others = dataloader_train.next()

			def loss(model, question, image, trees, y):
				model.structured_input_fn(trees)
				prediction, node_values = model((question, image))
				log_loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)
				
				# display
				if opt.logdir is not None:
					tf.contrib.summary.scalar('loss', log_loss)
					if opt.display_att == True:
						mvizer.show_node_values(node_values[1], others[3], others[1], inputs[0].cpu(), labels, predicts)
				return log_loss

			optimizer.minimize(lambda: loss(net, question, image, trees, labels),
				global_step=tf.train.get_or_create_global_step())
			global_step.assign_add(1)

def test(epoch):

	print('Evaluating...')
	dataloader_test.reset_state()

	avg_loss = tfe.metrics.Mean('loss')
	accuracy = tfe.metrics.Accuracy('accuracy')

	for _ in tqdm(range(dataloader_test.num_iter)):
		question, image, trees, labels, others = dataloader_test.next()

		# do forward
		net.structured_input_fn(trees)
		prediction, node_values = net((question, image))
		avg_loss(loss(prediction, labels))
		accuracy(
			tf.argmax(prediction, axis=1, output_type=tf.int64),
			tf.cast(labels, tf.int64))

	print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
		(avg_loss.result(), 100 * accuracy.result()))

	if opt.logdir is not None:
		with tf.contrib.summary.always_record_summaries():
			tf.contrib.summary.scalar('accuracy', accuracy.result())
			tf.contrib.summary.scalar('test_loss', avg_loss.result())

def checkpoint(epoch):
    checkpoint_prefix = os.path.join(opt.logdir, 'model_epoch_{}'.format(epoch))
    root.save(file_prefix=checkpoint_prefix)
    print('Checkpoint saved to {}'.format(checkpoint_prefix))

with tf.device(device):
	for epoch in range(finished_epoch + 1, finished_epoch + opt.max_epoch + 1):
		train(epoch)
		checkpoint(epoch)
		test(epoch)