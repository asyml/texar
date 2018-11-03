#!/usr/bin/env python3
# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attentional Seq2seq.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle

try:
    import queue
except ImportError:
    import Queue as queue

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

__all__ = [
    "Trigger",
    "BestEverConvergenceTrigger",
]


DEFAULT_ACTION = object()


class Trigger(object):
    """This is the base class of all triggers. A trigger can do some action when
    certain condition is met. Specifically, the user calls the trigger
    periodically. Every time the trigger is called, it will send all arguments
    to :meth:`_predicate`, which returns a boolean value indicates whether the
    condition is met. Once the condition is met, the trigger will then call
    `next(action)` to do next action and obtain the returned value.

    Args:
        action (iterable): An iterable which iteratively does the action and
            possibly returns a value.
        default (optional): The value returned after :attr:`action` exhausted.
            If not provided, the trigger will do nothing when `StopIteration`
            occurs.
    """

    def __init__(self, action, default=DEFAULT_ACTION):
        self._action = iter(action)
        self._default = default
        self._triggered_times = 0

    def _predicate(self, *args, **kwargs):
        """This function returns True when the condition is met and we should
        do something.
        """
        raise NotImplementedError

    def _next_action(self):
        return next(self._action) if self._default is DEFAULT_ACTION else \
               next(self._action, self._default)

    def __call__(self, *args, **kwargs):
        pred = self._predicate(*args, **kwargs)
        if pred:
            ret = self._next_action()
            self._triggered_times += 1
        else:
            ret = None
        return pred, ret

    def _make_state(self, names):
        return {name: getattr(self, name) for name in names}

    @property
    def _state_names(self):
        """Returns a list of names of attributes of the trigger object that can
        be saved and restored as trigger state.
        """
        return ['_triggered_times']

    @property
    def state(self):
        """The current state which can be used to save and restore the trigger.
        The state records how many times `next(action)` has been called.
        """
        return self._make_state(self._state_names)

    def restore_from_state(self, state):
        """Restore the trigger state from the previous stored state.
        Note that this function will call `next(action)` for the exact times
        that the :py:attr:`state` records how many times `next(action)` had
        been called. The user should be aware of any possible side effect of
        this behavior.

        Args:
            state: The state previously obtained by :py:attr:`state`.
        """
        for name, value in state.items():
            setattr(self, name, value)

        for t in range(self._triggered_times):
            self._next_action()

    def save_to_pickle(self, file):
        """Write a pickled representation of the state of the trigger to the
        open file-like object :attr:`file`.

        Args:
            file: The open file-like object to which we write. As described in
                pickle official document, it must have a `write()` method that
                accepts a single string argument.
        """
        pickle.dump(self.state, file)

    def restore_from_pickle(self, file):
        """Read a string from the open file-like object :attr:`file` and
        restore the trigger state from it.
        Note that this function will call `next(action)` for the exact times
        that the :py:attr:`state` records how many times `next(action)` had
        been called. The user should be aware of any possible side effect of
        this behavior.

        Args:
            file: The open file-like object from which we read. As described in
                pickle official document, it must have a `read()` method that
                takes an integer argument, and a `readline()` method that
                requires no arguments, and both methods should return a string.
        """
        self.restore_from_state(pickle.load(file))


class ScheduledStepsTrigger(Trigger):
    
    def __init__(self, action, steps, default=DEFAULT_ACTION):
        """steps should be in increasing order.
        """
        super(ScheduledTrigger, self).__init__(action, default)
        self._steps = iter(steps)
        self._advance_steps()

    def _advance_steps(self):
        self._next_step = next(step, None)

    def _predicate(self, step):
        while self._next_step is not None and step < self._next_step:
            self._advance_steps()
        if self._next_step is not None and step == self._next_step:
            return True
        return False


class BestEverConvergenceTrigger(Trigger):
    """A trigger that maintains the best value of a metric. It triggers when
    the best value of the metric has not been updated for at least
    :attr:`threshold_steps`. In order to avoid it triggers two frequently, it
    will not trigger again within :attr:`minimum_interval_steps` once it
    triggers.

    Args:
        action (iterable): An iterable which iteratively does the action and
            possibly returns a value.
        threshold_steps (int): Number of steps it should trigger after the best
            value was last updated.
        minimum_interval_steps (int): Minimum number of steps between twice
            firing of the trigger.
        default (optional): The value returned after :attr:`action` exhausted.
            If not provided, the trigger will do nothing when `StopIteration`
            occurs.
    .. document private functions
    .. automethod:: __call__
    """

    def __init__(self, action, threshold_steps, minimum_interval_steps,
                 default=DEFAULT_ACTION):
        super(BestEverConvergenceTrigger, self).__init__(action, default)
        self._threshold_steps = threshold_steps
        self._minimum_interval_steps = minimum_interval_steps
        self._last_triggered_step = None
        self._best_ever_step = None
        self._best_ever_score = None

    def _predicate(self, step, score):
        if self._best_ever_score is None or self._best_ever_score < score:
            self._best_ever_score = score
            self._best_ever_step = step

        if (self._last_triggered_step is None or
                step - self._last_triggered_step >=
                self._minimum_interval_steps) and \
                step - self._best_ever_step >= self._threshold_steps:
            self._last_triggered_step = step
            return True
        return False

    def __call__(self, step, score):
        """The trigger must be called to update the current training step
        (:attr:`step`) and the current value of the maintained metric
        (:attr:`score`).

        Args:
            step (int): Current training step to update. The training step must
                be updated in ascending order.
            score (float): Current value of the maintained metric.

        Returns:
            A tuple `(triggered, retval)`, where boolean `triggered` denotes
            whether triggered this time and `retval` is the return value of the
            action performed this time.
        """
        return super(BestEverConvergenceTrigger, self).__call__(step, score)

    @property
    def _state_names(self):
        return super(BestEverConvergenceTrigger, self)._state_names + [
            '_last_triggered_step', '_best_ever_step', '_best_ever_score']

    @property
    def last_triggered_step(self):
        """The step at which the Trigger last triggered.
        """
        return self._last_triggered_step

    @property
    def best_ever_step(self):
        """The step at which the best-ever score is reached.
        """
        return self._best_ever_step

    @property
    def best_ever_score(self):
        """The best-ever score.
        """
        return self._best_ever_score


class MovingAverageConvergenceTrigger(Trigger):

    def __init__(self, action, n, threshold, minimum_interval_steps,
                 default=DEFAULT_ACTION):
        super(MovingAverageConvergenceTrigger, self).__init__(action, default)
        self._n = n
        self._threshold = threshold
        self._minimum_interval_steps = minimum_interval_steps
        self._last_triggered_step = None
        self._head_queue = queue.Queue(self._n)
        self._head_sum = 0
        self._rear_queue = queue.Queue(self._n)
        self._rear_sum = 0

    def _predicate(self, step, score):
        if self._head_queue.full():
            e = self._head_queue.get()
            self._head_sum -= e
            if self._rear_queue.full():
                self._rear_sum -= self._rear_queue.get()
            self._rear_queue.put(e)
            self._rear_sum += e
        self._head_queue.put(score)
        self._head_sum += score

        if (self._last_triggered_step is None or
                step - self._last_triggered_step
                >= self._minimum_interval_steps) and \
                self._head_queue.full() and self._rear_queue.full() and \
                self._head_sum - self._rear_sum <= self._n * self._threshold:
            self._last_triggered_step = step
            return True
        return False

    @property
    def _state_names(self):
        return super(BestEverConvergenceTrigger, self)._state_names + [
            '_last_triggered_step', '_head_queue', '_head_sum', '_rear_queue',
            '_rear_sum']
