from argparse import ArgumentParser

class Config():
    """
        configuration for preprocessing.
    """
    #pylint:disable=attribute-defined-outside-init
    #pylint:disable=too-few-public-methods
    #pylint:disable=too-many-instance-attributes
    def __init__(self):
        self.input_dir = None
        self.src, self.tgt = None, None
        self.max_seq_length = None
def get_preprocess_args():
    """Data preprocessing options."""
    config = Config()
    parser = ArgumentParser(description='Preprocessing Options')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--tok', dest='tok', action='store_true',
                        help='tokenized and lowercased')
    parser.set_defaults(tok=False)
    parser.add_argument('--max_seq_length', type=int, default=70)
    parser.add_argument('--pre_encoding', type=str, default='spm')
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--tgt', type=str, default='vi')
    parser.add_argument('--input_dir', '-i', type=str, \
        default='./data/en_vi/data/', help='Input directory')
    parser.add_argument('--save_data', type=str, default='preprocess', \
        help='Output file for the prepared data')
    parser.parse_args(namespace=config)

    #keep consistent with original implementation
    #pylint:disable=attribute-defined-outside-init
    config.input = config.input_dir
    config.source_train = 'train.' + config.src
    config.target_train = 'train.' + config.tgt
    config.source_valid = 'valid.' + config.src
    config.target_valid = 'valid.' + config.tgt
    config.source_test = 'test.'+ config.src
    config.target_test = 'test.' + config.tgt
    return config
