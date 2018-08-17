__author__ = 'max'


class CoNLLWriter(object):
    def __init__(self, i2w, i2n):
        self.__source_file = None
        self.__i2w = i2w
        self.__i2n = i2n

    def start(self, file_path):
        self.__source_file = open(file_path, 'w', encoding='utf-8')

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__i2w[word[i, j]]
                tgt = self.__i2n[targets[i, j]]
                pred = self.__i2n[predictions[i, j]]
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, "_", "_", tgt, pred))
            self.__source_file.write('\n')
