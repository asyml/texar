import numpy as np
import preprocess
from chainer.dataset import convert

def to_cpu(x):
    try:
        y = x.data.cpu().tolist()[0]
    except:
        y = x.data.cpu().tolist()
    return y

def seq2seq_pad_concat_convert(xy_batch, device=-1,
    eos_id=preprocess.Vocab_Pad.EOS,
    bos_id=preprocess.Vocab_Pad.BOS):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.
    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with 0 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)
    x_block = convert.concat_examples(x_seqs, device, padding=0)
    y_block = convert.concat_examples(y_seqs, device, padding=0)

    # Add EOS
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant',
        constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    y_out_block = np.pad(y_block, ((0, 0), (0, 1)), 'constant',
        constant_values=0)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    # Add BOS in target language
    y_in_block = np.pad(y_block, ((0, 0), (1, 0)), 'constant',
        constant_values=bos_id)
    return x_block, y_in_block, y_out_block

def source_pad_concat_convert(x_seqs,
    eos_id=preprocess.Vocab_Pad.EOS,
    bos_id=preprocess.Vocab_Pad.BOS):
    x_block = convert.concat_examples(x_seqs, device=-1, padding=0)
    """
    This function is used when testing the model without target input.
    """
    # add eos
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    return x_block
