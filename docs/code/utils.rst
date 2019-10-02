.. role:: hidden
    :class: hidden-section

Utils
**************

Frequent Use
============

:hidden:`AverageRecorder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.tf.utils.AverageRecorder
    :members:

:hidden:`collect_trainable_variables`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.collect_trainable_variables

:hidden:`compat_as_text`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.compat_as_text

:hidden:`map_ids_to_strs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.map_ids_to_strs

:hidden:`write_paired_text`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.write_paired_text

:hidden:`straight_through`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.straight_through


Variables
=========

:hidden:`collect_trainable_variables`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.collect_trainable_variables

:hidden:`get_unique_named_variable_scope`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_unique_named_variable_scope

:hidden:`add_variable`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.add_variable


IO
===

:hidden:`write_paired_text`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.write_paired_text

:hidden:`load_config`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.load_config

:hidden:`maybe_create_dir`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.maybe_create_dir

:hidden:`get_files`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_files


DType
=====

:hidden:`compat_as_text`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.compat_as_text

:hidden:`get_tf_dtype`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_tf_dtype

:hidden:`is_callable`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_callable

:hidden:`is_str`
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_str

:hidden:`is_placeholder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_placeholder

:hidden:`maybe_hparams_to_dict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.maybe_hparams_to_dict


Shape
=====

:hidden:`mask_sequences`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.mask_sequences

:hidden:`transpose_batch_time`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.transpose_batch_time

:hidden:`get_batch_size`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_batch_size

:hidden:`get_rank`
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_rank

:hidden:`shape_list`
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.shape_list

:hidden:`pad_and_concat`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.pad_and_concat

:hidden:`reduce_with_weights`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.reduce_with_weights

:hidden:`flatten`
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.flatten

:hidden:`varlength_concat`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.varlength_concat

:hidden:`varlength_concat_py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.varlength_concat_py

:hidden:`varlength_roll`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.varlength_roll


Dictionary
===========

:hidden:`dict_patch`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.dict_patch

:hidden:`dict_lookup`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.dict_lookup

:hidden:`dict_fetch`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.dict_fetch

:hidden:`dict_pop`
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.dict_pop

:hidden:`flatten_dict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.flatten_dict

String
=======

:hidden:`map_ids_to_strs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.map_ids_to_strs

:hidden:`strip_token`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.strip_token

:hidden:`strip_eos`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.strip_eos

:hidden:`strip_special_tokens`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.strip_special_tokens

:hidden:`str_join`
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.str_join

:hidden:`default_str`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.default_str

:hidden:`uniquify_str`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.uniquify_str


Meta
====

:hidden:`check_or_get_class`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.check_or_get_class

:hidden:`get_class`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_class

:hidden:`check_or_get_instance`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.check_or_get_instance

:hidden:`get_instance`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_instance

:hidden:`check_or_get_instance_with_redundant_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.check_or_get_instance_with_redundant_kwargs

:hidden:`get_instance_with_redundant_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_instance_with_redundant_kwargs

:hidden:`get_function`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_function

:hidden:`call_function_with_redundant_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.call_function_with_redundant_kwargs

:hidden:`get_args`
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_args

:hidden:`get_default_arg_values`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_default_arg_values

:hidden:`get_instance_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.get_instance_kwargs


Mode
====

:hidden:`switch_dropout`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.switch_dropout

:hidden:`maybe_global_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.maybe_global_mode

:hidden:`is_train_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_train_mode

:hidden:`is_eval_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_eval_mode

:hidden:`is_predict_mode`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_predict_mode

:hidden:`is_train_mode_py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_train_mode_py

:hidden:`is_eval_mode_py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_eval_mode_py

:hidden:`is_predict_mode_py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.is_predict_mode_py

Misc
====

:hidden:`ceildiv`
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.ceildiv

:hidden:`straight_through`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.straight_through

:hidden:`truncate_seq_pair`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.tf.utils.truncate_seq_pair

AverageRecorder
==========================
.. autoclass:: texar.tf.utils.AverageRecorder
    :members:
