# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from PIL import Image
from tfrecords_data import TFRecordData

class TFRecordDataTest(tf.test.TestCase):
    """Tests scalar data class.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)

        # Create test data
        # pylint: disable=no-member
        cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
        williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
        image_labels = {
            cat_in_snow : 0,
            williamsburg_bridge : 1,
        }
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value]))
        self._dataset_valid = {
            'height': [],
            'width': [],
            'label': [],
            'image_raw': [],
        }
        def _construct_dataset_valid(image, label):
            image_string = image.tostring()
            single_data = {
                'height': image.shape[0],
                'width': image.shape[1],
                'label': label,
                'image_raw': image_string,
            }
            for key, value in single_data.items():
                self._dataset_valid[key].append(value)
        self._feature_key_and_types = {
            'height': 'tf.int64',
            'width': 'tf.int64',
            'label': 'tf.int64',
            'image_raw': 'tf.string'
        }
        def _image_example(image, label):
            image_string = image.tostring()
            image_shape = image.shape
            feature = {
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(image_string),
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))

        self.tfrecords_filename = 'images.tfrecords'
        with tf.python_io.TFRecordWriter(self.tfrecords_filename) as writer:
            for filename, label in image_labels.items():
                image = np.array(Image.open(filename))
                tf_example = _image_example(image, label)
                _construct_dataset_valid(image, label)
                writer.write(tf_example.SerializeToString())

        self._img_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self.tfrecords_filename,
                "feature_key_and_types": self._feature_key_and_types
            }
        }



    def _run_and_test(self, hparams):
        # Construct database
        expm_img_data = TFRecordData(hparams)
        iterator = expm_img_data.dataset.make_initializable_iterator()
        data_batch = iterator.get_next()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            i = 0
            while True:
                try:
                    # Run the logics
                    data_batch_ = sess.run(data_batch)
                    self.assertEqual(set(data_batch_.keys()),
                                     set(expm_img_data.list_items()))
                    for key in data_batch_['data'].keys():
                        value = data_batch_['data'][key][0]
                        self.assertEqual(value, self._dataset_valid[key][i])
                    i += 1
                except tf.errors.OutOfRangeError:
                    print('Done -- epoch limit reached')
                    break

    def test_default_setting(self):
        """Tests the logics of ImageData.
        """
        self._run_and_test(self._img_hparams)

if __name__ == "__main__":
    tf.test.main()
