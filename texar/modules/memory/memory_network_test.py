"""
Unit tests for memory networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.memory.memory_network import MemNetRNNLike
from texar import context

# pylint: disable=no-member, too-many-locals, too-many-instance-attributes
# pylint: disable=too-many-arguments, protected-access

class MemNetRNNLikeTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.memory.memory_network.MemNetRNNLike`.
    """

    def _test_memory_dim(self, combine_mode='add', soft_memory=False,
                         soft_query=False, use_B=False):
        """Tests :attr:`memory_dim` in the :attr:`combine_mode` and soft
        options.
        """
        print('testing: combine_mode={}, soft_memory={}, soft_query={}, '
              'use_B={}'.format(combine_mode, soft_memory, soft_query, use_B))

        n_hops = 3
        if combine_mode == 'add' or combine_mode is None:
            memory_dim = 19
            embedding_dim = memory_dim
            temporal_embedding_dim = memory_dim
        elif combine_mode == 'concat':
            embedding_dim = 19
            temporal_embedding_dim = 17
            memory_dim = embedding_dim + temporal_embedding_dim
        else:
            raise ValueError(
                "combine_mode = {} is not recognized".format(combine_mode))
        relu_dim = 13
        memory_size = 7
        raw_memory_dim = 11
        batch_size = 2
        embed_hparams = {
            "embedding": {
                "dim": embedding_dim,
            },
            "temporal_embedding": {
                "dim": temporal_embedding_dim,
            },
            "combine_mode": combine_mode,
        }
        memnet_hparams = {
            "n_hops": n_hops,
            "relu_dim": relu_dim,
            "memory_size": memory_size,
            "A": embed_hparams,
            "C": embed_hparams,
            "B": embed_hparams,
            "use_B": use_B,
        }
        
        memnet = MemNetRNNLike(raw_memory_dim=raw_memory_dim,
                               hparams=memnet_hparams)
        kwargs = {}
        if soft_memory:
            kwargs['soft_memory'] = tf.random_uniform(
                [batch_size, memory_size, raw_memory_dim])
        else:
            kwargs['memory'] = tf.tile(tf.expand_dims(
                tf.range(memory_size, dtype=tf.int32), 0), [batch_size, 1])
        if use_B:
            if soft_query:
                kwargs['soft_query'] = tf.random_uniform(
                    [batch_size, raw_memory_dim])
            else:
                kwargs['query'] = tf.random_uniform(
                    [batch_size], maxval=raw_memory_dim, dtype=tf.int32)
        else:
            kwargs['query'] = tf.random_uniform([batch_size, memory_dim])
        logits = memnet(**kwargs)
        self.assertEqual(memnet.memory_dim, memory_dim)
        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], raw_memory_dim)

    def test_memory_dim(self):
        """Tests :attr:`memory_dim` in different :attr:`combine_mode` and
        different soft options.
        """
        for combine_mode in ['add', 'concat']:
            for soft_memory in [False, True]:
                for use_B in [False, True]:
                    for soft_query in ([False, True] if use_B else [False]):
                        self._test_memory_dim(combine_mode, soft_memory,
                                              soft_query, use_B)

if __name__ == "__main__":
    tf.test.main()
