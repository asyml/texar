#
"""
Utility functions related to input/output.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from io import open # pylint: disable=redefined-builtin
#import logging

import tensorflow as tf
as_text = tf.compat.as_text

__all__ = [
    "write_paired_text"
]

#def get_tf_logger(fname,
#                  verbosity=tf.logging.INFO,
#                  to_stdio=False,
#                  stdio_verbosity=None):
#    """Creates TF logger that allows to specify log filename and whether to
#    print to stdio at the same time.
#
#    Args:
#        fname (str): The log filename.
#        verbosity: The threshold for what messages will be logged. Default is
#            `INFO`. Other options include `DEBUG`, `ERROR`, `FATAL`, and `WARN`.
#            See :tf_main:`tf.logging <logging>`.
#        to_stdio (bool): Whether to print messages to stdio at the same time.
#        stido_verbosity (optional): The verbosity level when printing to stdio.
#            If `None` (default), the level is set to be the same as
#            :attr:`verbosity`. Ignored if :attr:`to_stdio` is False.
#
#    Returns:
#        The TF logger.
#    """



def write_paired_text(src, tgt, fname, append=False, mode='h', sep='\t'):
    """Writes paired text to a file.

    Args:
        src: A list (or array) of `str` source text.
        ttg: A list (or array) of `str` target text.
        fname (str): The output filename.
        append (bool): Whether appending to the end of the file if exists.
        mode (str): The mode of writing, with the following options:

            - :attr:`'h'`: The "horizontal" mode. Each source target pair is \
                written in one line, intervened with :attr:`sep`, e.g.,

                    source_1 target_1
                    source_2 target_2

            - :attr:`'v'`: The "vertical" mode. Each source target pair is \
                written in two consecutive lines, e.g,

                    source_1
                    target_1
                    source_2
                    target_2

            - :attr:`'s'`: The "separate" mode. Each source target pair is \
                    written in corresponding lines of two files named \
                    as :attr:`fname`.src and :attr:`fname`.tgt, respectively.
        sep (str): The string intervening between source and target. Used
            when :attr:`mode`='h'.

    Returns:
        The fileanme(s). If :attr:`mode`=='h' or :attr:`mode`=='v', returns
        :attr:`fname`. If :attr:`mode`=='s', returns a list of filenames
        `[':attr:`fname`.src', ':attr:`fname`.tgt']`.
    """
    fmode = 'a' if append else 'w'
    if mode == 's':
        fn_src = '{}.src'.format(fname)
        fn_tgt = '{}.tgt'.format(fname)
        with open(fn_src, fmode, encoding='utf-8') as fs:
            fs.write(as_text('\n'.join(src)))
            fs.write('\n')
        with open(fn_tgt, fmode, encoding='utf-8') as ft:
            ft.write(as_text('\n'.join(tgt)))
            ft.write('\n')
        return fn_src, fn_tgt
    else:
        with open(fname, fmode, encoding='utf-8') as f:
            for s, t in zip(src, tgt):
                if mode == 'h':
                    text = '{}{}{}\n'.format(as_text(s), sep, as_text(t))
                    f.write(as_text(text))
                elif mode == 'v':
                    text = '{}\n{}\n'.format(as_text(s), as_text(t))
                    f.write(as_text(text))
                else:
                    raise ValueError('Unknown mode: {}'.format(mode))
        return fname
