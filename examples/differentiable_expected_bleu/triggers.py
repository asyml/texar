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

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

try:
    import queue
except ImportError:
    import Queue as queue

DEFAULT = object()

class Trigger(object):

    def __init__(self, action, default=DEFAULT):
        """action is an iterator that iteratively do a sequence of action and
        return result values. default is used as result value when action is
        exhausted.
        """
        self._action = action
        self._default = default

    def predicate(self, *args, **kwargs):
        """This function returns True when we think we should do something.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        pred = self.predicate(*args, **kwargs)
        if pred:
            ret = next(self._action) if self._default is DEFAULT else \
                  next(self._action, self._default)
        else:
            ret = None
        return pred, ret


class ScheduledStepsTrigger(Trigger):
    
    def __init__(self, action, steps, default=DEFAULT):
        """steps should be in increasing order.
        """
        super(ScheduledTrigger, self).__init__(action, default)
        self._steps = iter(steps)
        self._advance_steps()

    def _advance_steps(self):
        self._next_step = next(step, None)

    def predicate(self, step):
        while self._next_step is not None and step < self._next_step:
            self._advance_steps()
        if self._next_step is not None and step == self._next_step:
            return True
        return False


class BestEverConvergenceTrigger(Trigger):

    def __init__(self, action, threshold_steps, wait_steps, default=DEFAULT):
        super(BestEverConvergenceTrigger, self).__init__(action, default)
        self._threshold_steps = threshold_steps
        self._wait_steps = wait_steps
        self._last_triggered_step = None
        self._best_ever_step = None
        self._best_ever_score = None

    def predicate(self, step, score):
        if self._best_ever_score is None or self._best_ever_score < score:
            self._best_ever_score = score
            self._best_ever_step = step

        if (self._last_triggered_step is None or
                step - self._last_triggered_step >= self._wait_steps) and \
                step - self._best_ever_step >= self._threshold_steps:
            self._last_triggered_step = step
            return True
        return False


class MovingAverageConvergenceTrigger(Trigger):

    def __init__(self, action, n, threshold, wait_steps, default=DEFAULT):
        super(MovingAverageConvergenceTrigger, self).__init__(action, default)
        self._n = n
        self._threshold = threshold
        self._wait_steps = wait_steps
        self._last_triggered_step = None
        self._head_queue = queue.Queue(self._n)
        self._head_sum = 0
        self._rear_queue = queue.Queue(self._n)
        self._rear_sum = 0

    def predicate(self, step, score):
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
                step - self._last_triggered_step >= self._wait_steps) and \
                self._head_queue.full() and self._rear_queue.full() and \
                self._head_sum - self._rear_sum <= self._n * self._threshold:
            self._last_triggered_step = step
            return True
        return False
