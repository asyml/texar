import codecs

def write_words(words_list, filename):
    with codecs.open(filename, 'w+', 'utf-8') as myfile:
        for words in words_list:
            myfile.write(' '.join(words) + '\n')

