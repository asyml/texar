import numpy as np
import os
import six

def load_data_numpy(input_dir, prefix):
    train_data = np.load(os.path.join(input_dir,\
        prefix + 'train.npy')).tolist()
    dev_data = np.load(os.path.join(input_dir,\
        prefix + 'valid.npy')).tolist()
    test_data = np.load(os.path.join(input_dir,\
        prefix + 'test.npy')).tolist()
    print('train data size:{}'.format(len(train_data)))
    return train_data, dev_data, test_data

def seq2seq_pad_concat_convert(xy_batch, device=-1,
    eos_id=2,
    bos_id=1):
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
        eos_id: The index of end-of-sentence special token in the
            dictionary.
    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with 0 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)
    x_block = _concat_examples(x_seqs, device, padding=0)
    y_block = _concat_examples(y_seqs, device, padding=0)

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
    eos_id=2,
    bos_id=1):
    x_block = _concat_examples(x_seqs, device=-1, padding=0)
    """
    This function is used when testing the model without target input.
    """
    # add eos
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    return x_block


def _concat_examples(arrays, device=-1, padding=0):
    if len(arrays) == 0:
        raise ValueError('batch is empty')

    first_elem = arrays[0]
    assert isinstance(first_elem, np.ndarray)

    shape = np.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if np.any(shape != array.shape):
            np.maximum(shape, array.shape, shape)
    shape = tuple(np.insert(shape, 0, len(arrays)))

    result = np.full(shape, padding, dtype=arrays[0].dtype)
    for i in six.moves.range(len(arrays)):
        src = arrays[i]
        slices = tuple(slice(dim) for dim in src.shape)
        result[(i,) + slices] = src
    return result

if __name__ == "__main__":
    pass
