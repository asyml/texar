# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" loader for switch board dataset.
"""
import os
import json
from json_lines import reader

from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import texar as tx

from config_data import data_root

# pylint: disable=invalid-name, too-many-locals

wnd_sz = 10

class Dataset(object):
    """Data preprocessor.
    """

    def __init__(self, jsonl_path, mode=None):
        self.mode = mode
        self.raw = []
        self.lst = []
        self.refs = []
        if mode == 'test':
            lst = json.load(open(jsonl_path, 'r'))
            for item in lst:
                context = item['context']
                dialog = []
                for utts in context:
                    p = utts.find(':')
                    dialog.append((
                        (utts[p-1] == 'A') * 2 - 1, utts[p + 2:-1], 0))

                if dialog[0][1][-1] == '>':
                    dialog = dialog[1:]

                if len(dialog) == 0:
                    continue

                responses = []
                for resp in item['responses']:
                    responses.append(resp)

                spk = (item['speaker'] == 'A') * 2 - 1
                dialog.append((spk, responses[0], 0))
                responses = responses[1:]
                responses = [' '.join(WordPunctTokenizer().tokenize(resp))
                             for resp in responses]

                if len(responses) == 0:
                    continue

                self.raw.append(dialog)
                self.lst.append((len(self.raw) - 1, 0, len(dialog)))
                self.refs.append(responses)

            return

        from collections import Counter
        self.ct = Counter()
        self.topics = []
        with open(jsonl_path, 'r') as f:
            for idx, item in enumerate(reader(f)):
                utts = item['utts']
                self.topics.append(item['topic'])
                self.raw.append([(int(speaker == 'A') * 2 - 1, sentence, _)
                                 for speaker, sentence, _ in utts])

                lst = [(idx, start, start + wnd_sz)
                       for start in range(0, len(utts)-wnd_sz)] + \
                      [(idx, 0, end)
                       for end in range(2, min(wnd_sz+1, len(utts)))]

                self.lst += lst

        self.refs = [['none']] * len(self.lst)

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        idx, start, end = self.lst[idx]
        dialog = self.raw[idx][start:end]
        source, target = dialog[:-1], dialog[-1]

        spks, utts = list(zip(*[(speaker, WordPunctTokenizer().tokenize(uttr)) for speaker, uttr, _ in source]))

        spks = list(spks)

        while len(spks) < 10:
            spks.append(0)

        source = '|||'.join([' '.join(uttr) for uttr in utts])
        target_test = ' '.join(WordPunctTokenizer().tokenize(target[1]))

        return spks, source, target_test, target[0]

    def get(self, idx):
        idx, start, end = self.lst[idx]
        source = self.raw[idx][start:end-1]
        target = self.raw[idx][end-1]
        source = ' '.join([b for a, b, c in source])
        cct = self.raw[idx][end-2][0] == self.raw[idx][end-1][0]
        return self.topics[idx], cct, source, target

def sw1c2r(data_root):
    dts_train = Dataset(os.path.join(data_root, 'train.jsonl'))
    dts_valid = Dataset(os.path.join(data_root, 'valid.jsonl'))
    dts_test = Dataset(os.path.join(data_root, 'test_multi_ref.json'), 'test')
    datasets = {
        'train': dts_train,
        'val': dts_valid,
        'test': dts_test
    }
    return datasets

def generate_reference_for_test_dialog(dataset, data_root):
    vocab = {}
    with open(os.path.join(data_root, 'vocab.txt'), 'r') as f:
        p = f.read().splitlines()
        for i, x in enumerate(p):
            vocab[x] = i

    dts_train = dataset['train']
    dts_val = dataset['val']
    dts_test = dataset['test']

    vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize,
                                 vocabulary=vocab)

    saved = []
    meta = []
    data = []
    tidx = {}
    for i in range(len(dts_test)):
        topic, cct, source, target = dts_test.get(i)
        meta.append((topic, cct, target))
        data.append(source)

    for i in range(len(dts_train)):
        topic, cct, source, target = dts_train.get(i)
        saved.append((topic, cct, target))
        data.append(source)

        if topic not in tidx:
            tidx[topic] = []
        tidx[topic].append(i)

    result = vectorizer.fit_transform(data)
    x = result[:len(dts_test)]
    y = result[len(dts_test):]

    from tqdm import tqdm
    from sklearn.preprocessing import normalize

    y = normalize(y)
    x = normalize(x)

    dts_test.refs = []
    for i in tqdm(range(len(dts_test))):
        c = tidx[meta[i][0]]
        p = (y * x[i].T).toarray().reshape(-1)[c]
        d = p.argsort()

        cnt = 0
        refs = []
        for a in d[::-1]:
            if saved[a][1] == meta[i][1]:
                refs.append(' '.join(
                    WordPunctTokenizer().tokenize(saved[a][2][1])))
                cnt += 1
                if cnt == 10:
                    break

        dts_test.refs.append(refs)

def download_and_process(data_root):
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
        os.makedirs(os.path.join(data_root, 'raw'))

        tx.data.maybe_download(
            urls='https://drive.google.com/file/d/1Gytd-SSetUkIY6aVVKNrBOxkHjAlSGeU/view?usp=sharing',
            path='./',
            filenames=os.path.join(data_root, 'sw1c2r.tar.gz'),
            extract=True)

        os.system('mv {} {}'.format(os.path.join(data_root, 'sw1c2r.tar.gz'),
                                    os.path.join(data_root, 'raw/sw1c2r.tar.gz')))
        os.system('mv {}/* {}'.format(
            os.path.join(data_root, 'switchboard'), data_root))

        datasets = sw1c2r(os.path.join(data_root, 'json_data'))

        for stage in ['train', 'val', 'test']:
            dts = datasets[stage]
            spk, src, tgt, meta = list(zip(*[dts[i] for i in range(len(dts))]))
            src_txt = '\n'.join(src)
            tgt_txt = '\n'.join(tgt)

            spk = list(zip(*spk))

            for i in range(len(spk)):
                with open(os.path.join(data_root, '{}-source-spk-{}.txt'.format(stage, i)), 'w') as f:
                    f.write('\n'.join([str(a) for a in spk[i]]))

            spk_tgt = meta

            with open(os.path.join(data_root, '{}-target-spk.txt'.format(stage)), 'w') as f:
                f.write('\n'.join([str(a) for a in spk_tgt]))

            with open(os.path.join(data_root, '{}-source.txt'.format(stage)), 'w') as f:
                f.write(src_txt)
            with open(os.path.join(data_root, '{}-target.txt'.format(stage)), 'w') as f:
                f.write(tgt_txt)

            with open(os.path.join(data_root, '{}-target-refs.txt'.format(stage)), 'w') as f:
                f.write('\n'.join(['|||'.join(v) for v in dts.refs]))

if __name__ == '__main__':
    download_and_process(data_root)
