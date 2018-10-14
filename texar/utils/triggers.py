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

DEFAULT_ACTION = object()

class Trigger(object):

    def __init__(self, action, default=DEFAULT_ACTION):
        """action is an iterator that iteratively do a sequence of action and
        return result values. default is used as result value when action is
        exhausted.
        """
        self._action = iter(action)
        self._default = default
        self._triggered_times = 0

    def _predicate(self, *args, **kwargs):
        """This function returns True when we think we should do something.
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
        return ['_triggered_times']

    @property
    def state(self):
        return self._make_state(self._state_names)

    def restore_from_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        for t in range(self._triggered_times):
            self._next_action()

    def save_to_pickle(self, file):
        pickle.dump(self.state, file)

    def restore_from_pickle(self, file):
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

    @property
    def _state_names(self):
        return super(BestEverConvergenceTrigger, self)._state_names + [
            '_last_triggered_step', '_best_ever_step', '_best_ever_score']


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
