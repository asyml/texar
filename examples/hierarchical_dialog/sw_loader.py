import os
from json_lines import reader

from nltk.tokenize import WordPunctTokenizer

import texar as tx

url = 'https://raw.githubusercontent.com/shyaoni/sw1c2r/master/sw1c2r.tar.gz'

WINDOW_SIZE = 10

class Dataset():
    def __init__(self, jsonl_path):
        self.raw = []
        self.lst = []
        with open(jsonl_path, 'r') as f:
            for idx, item in enumerate(reader(f)):
                utts = item['utts']
                self.raw.append([(int(speaker == 'A') * 2 - 1, sentence)
                                 for speaker, sentence, _ in utts]) 
                
                lst = [(idx, start, start + WINDOW_SIZE + 1)
                       for start in range(0, len(utts) - WINDOW_SIZE)] + \
                      [(idx, 0, end) for end in range(2, WINDOW_SIZE)]
                self.lst += lst 

    def __len__(self):
        return len(self.lst)
    
    def __getitem__(self, idx):
        idx, start, end = self.lst[idx]
        dialog = self.raw[idx][start:end]
        source, target = dialog[:-1], dialog[-1]

        utts = [WordPunctTokenizer().tokenize(uttr) for speaker, uttr in source]
        source = '|||'.join([' '.join(uttr) for uttr in utts])
        target = ' '.join(target[1])

        return source, target

def sw1c2r(data_root): 
    dts_train = Dataset(os.path.join(data_root, 'train.jsonl'))
    dts_valid = Dataset(os.path.join(data_root, 'valid.jsonl'))
    dts_test = Dataset(os.path.join(data_root, 'test.jsonl'))
    datasets = {
        'train': dts_train,
        'val': dts_valid,
        'test': dts_test
    }
    return datasets

def download_and_process(data_root):
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
        os.makedirs(os.path.join(data_root, 'raw'))

        tx.data.maybe_download([url], data_root, extract=True)

        os.system('mv {} {}'.format(os.path.join(data_root, 'sw1c2r.tar.gz'),
                                    os.path.join(data_root, 'raw/sw1c2r.tar.gz')))

        datasets = sw1c2r(os.path.join(data_root, 'json_data'))
    
        for stage in ['train', 'val', 'test']:
            dts = datasets[stage]
            src, tgt = list(zip(*[dts[i] for i in range(len(dts))]))
            src_txt = '\n'.join(src)
            tgt_txt = '\n'.join(tgt)

            with open(os.path.join(data_root, '{}-source.txt'.format(stage)), 'w') as f:
                f.write(src_txt)
            with open(os.path.join(data_root, '{}-target.txt'.format(stage)), 'w') as f:
                f.write(tgt_txt)

if __name__ == '__main__':
    download_and_process('test')
