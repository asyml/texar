#
"""Downloads data.
"""
import tensorflow as tf
import texar as tx

# pylint: disable=invalid-name

flags = tf.flags

flags.DEFINE_string("data", "iwslt14", "Data to download [iwslt14|toy_copy]")

FLAGS = flags.FLAGS

def prepare_data():
    """Downloads data.
    """
    if FLAGS.data == 'iwslt14':
        tx.data.maybe_download(
            urls='https://drive.google.com/file/d/'
                 '1Vuv3bed10qUxrpldHdYoiWLzPKa4pNXd/view?usp=sharing',
            path='./',
            filenames='iwslt14.zip',
            extract=True)
    elif FLAGS.data == 'toy_copy':
        tx.data.maybe_download(
            urls='https://drive.google.com/file/d/'
                 '1fENE2rakm8vJ8d3voWBgW4hGlS6-KORW/view?usp=sharing',
            path='./',
            filenames='toy_copy.zip',
            extract=True)
    else:
        raise ValueError('Unknown data: {}'.format(FLAGS.data))

def main():
    """Entrypoint.
    """
    prepare_data()

if __name__ == '__main__':
    main()
