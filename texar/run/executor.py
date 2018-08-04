#
"""
A class that executes training, evaluation, prediction, export of estimators.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.utils.dtypes import maybe_hparams_to_dict

class Executor(object):
    """Class that executes training, evaluation, prediction, export, and other
    actions of :tf_main:`Estimator <estimator/Estimator>`.

    Args:
        model: An instance of a subclass of
            :class:`~texar.models.model_base.ModelBase`.
        hparams: A `dict` or an instance of :class:`~texar.hparams.HParams`
            containing the hyperparameters of the model.
        data_hparams: A `dict` or an instance of :class:`~texar.hparams.HParams`
            containing the hyperparameters of data. It must contain `train`
            and/or `test` fields for relevant processes. For example, for
            :meth:`train_and_evaluate`, both fields are required.
        config: An instance of
            :tf_main:`tf.estimator.RunConfig <estimator/RunConfig>`, used as
            the :attr:`config` argument of
            :tf_main:`Estimator <estimator/Estimator>`.
        train_steps (int, optional): Total number of steps for which to train
            model. If `None`, train forever or unitil the train data generates
            the OutOfRange exception. See
            :tf_main:`Estimator.train <estimator/Estimator#train>` for
            more details.
        eval_steps (int, optional): Number of steps for which to evaluate
            model. If `None`, evaluates until the eval data raises an
            OutOfRange exception. See
            :tf_main:`Estimator.evaluate <estimator/Estimator#evaluate>` for
            more details.
        train_hooks (optional): Iterable of :tf_main:`tf.train.SessionRunHook
            <train/SessionRunHook>` objects to run during training.
        eval_hooks (optional): Iterable of :tf_main:`tf.train.SessionRunHook
            <train/SessionRunHook>` objects to run during evaluation.
        session_config (optional): An instance of
            :tf_main:`tf.ConfigProto <ConfigProto>`, used as the :attr:`config`
            argument of :tf_main:`tf session <Session>`.
    """

    def __init__(self,
                 model,
                 hparams,
                 data_hparams,
                 config,
                 train_steps=None,
                 eval_steps=None,
                 train_hooks=None,
                 eval_hooks=None,
                 session_config=None):
        self._model = model
        self._hparams = maybe_hparams_to_dict(hparams)
        self._data_hparams = maybe_hparams_to_dict(data_hparams)
        self._config = config
        self._train_steps = train_steps
        self._eval_steps = eval_steps
        self._train_hooks = train_hooks
        self._eval_hooks = eval_hooks
        self._session_config = session_config

        self._estimator = tf.estimator.Estimator(
            model_fn=self._model, config=config, params=self._hparams)

    def _get_train_spec(self):
        if 'train' not in self._data_hparams:
            raise ValueError('`data_hparams` must contain field `train` for '
                             'training processes.')
        input_fn = self._model.get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            hparams=self._data_hparams['train'])
        return tf.estimator.TrainSpec(
            input_fn=input_fn,
            max_steps=self._train_steps)

    #def _get_eval_spec(self):


    #def train(self):
    #    """Trains the model.
    #    """
    #    train_spec = self._build_train_spec()
    #    self._estimator.train(
    #        train_spec.input_fn, hooks=train_spec.hooks, max_steps=train_spec.max_steps)
    #    self._maybe_average_checkpoints()
