from __future__ import absolute_import, division, print_function

import tensorflow as tf
import h5py
import os.path
import json
import numpy as np
import math

def read_json(fname):
    file = open(fname, 'r')
    res = json.load(file)
    file.close()
    return res

class clevrDataset(object):
    """clevr dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            vocab_dir (string): Path to the h5 file of annotations.
            tree_dir (string): Path to the json file of dependency trees.
            img_h5_folder (string): Path to the h5 file with image features
            mode (string): Mode of train, val or test
        """
        self.tree_dir = opt.tree_dir
        self.vocab_dir = opt.vocab_dir
        self.img_h5_folder = opt.clevr_img_h5
        self.imgFolder = './CLEVR_v1.0/images' # TODO: used for read original image for visualization

        # qa h5
        if mode == 'train':
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_train.h5'), 'r')
            self.qas = {}
            self.qas['question'] = (np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = (np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = (np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            self.trees = read_json(os.path.join(self.tree_dir, 'clevr_train_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_train.h5', 'r')
        elif mode == 'val':
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_val.h5'), 'r')
            self.qas = {}
            self.qas['question'] = (np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = (np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = (np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            self.trees = read_json(os.path.join(self.tree_dir, 'clevr_val_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_val.h5', 'r')
        else:
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_test.h5'), 'r')
            self.qas = {}
            self.qas['question'] = (np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = (np.int64(file['/question_id_test'][:]))
            self.qas['img_id'] = (np.int32(file['/img_id_test'][:]))
            file.close()
            self.trees = read_json(os.path.join(self.tree_dir, 'clevr_test_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_test.h5', 'r')
        # train_test json
        vocab = read_json(os.path.join(self.vocab_dir, 'Vocab.json'))
        ansVocab = read_json(os.path.join(self.vocab_dir, 'AnsVocab.json'))
        opt.vocab_size = len(vocab)
        opt.out_vocab_size = len(ansVocab)

        opt.sent_len = self.qas['question'].shape[1]
        self.mode = mode
        num_samples = self.qas['question'].shape[0]
        if mode == 'train': self.num_iter = math.floor(num_samples / opt.batch_size)
        else: self.num_iter = math.ceil(num_samples / opt.batch_size)
        self.indices = np.arange(num_samples)
        self.cur_iter = 0
        self.batch_size = opt.batch_size

        print('    * clevr-%s loaded' % mode)

    
    def __getitem__(self, idx):
        img_id = self.qas['img_id'][idx]
        if self.mode == 'test': answer = None
        else:
            answer = self.qas['answers'][idx][0] - 1
            answer = answer.item()
        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):
            if self.mode == 'train': return self.imgFolder+'/train/CLEVR_train_%06d.png' % img_id
            elif self.mode == 'val': return self.imgFolder+'/val/CLEVR_val_%06d.png' % img_id
            else: return self.imgFolder+'/test/CLEVR_test_%06d.png' % img_id

        def load_image(img_name):
            img_name = os.path.basename(img_name)
            return (np.array(self.img_file[img_name]))
        
        img_name = id2imgName(img_id, qid)

        return self.qas['question'][idx], \
               qid, \
               answer, \
               load_image(img_name), \
               img_name, \
               self.trees[idx]

    def __len__(self):
        return self.qas['question'].shape[0]

    def num_iter(self):
        return self.num_iter

    def reset_state(self):
        self.cur_iter = 0
        np.random.shuffle(self.indices)

    def next(self):
        assert self.cur_iter < self.num_iter

        batch = []
        for i in range(self.cur_iter*self.batch_size, min((self.cur_iter+1)*self.batch_size, self.__len__()) ):
            batch.append(self.__getitem__(self.indices[i]))

        question, qid, answer, image, image_name, tree = zip(*batch)
        
        image = tf.stack(image)
        image = tf.transpose(image, perm=[0, 2, 3, 1]) # tranpose "channel first" to "channel last"
        question = tf.stack(question)
        answer = tf.convert_to_tensor(answer)
        others = qid, image_name

        self.cur_iter+=1
            
        return question, image, tree, answer, others


    
    
    
