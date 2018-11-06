"""
Unit tests for triggers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import bisect

from texar.utils.triggers import *


class TriggerTest(tf.test.TestCase):
    """Tests :class:`~texar.utils.Trigger`.
    """

    def test(self):
        trigger = Trigger(0, lambda x: x+1)
        for step in range(100):
            trigger.trigger()
            self.assertEqual(trigger.user_state, step+1)


class ScheduledStepsTriggerTest(tf.test.TestCase):
    """Tests :class:`~texar.utils.ScheduledStepsTrigger`.
    """

    def test(self):
        for i in range(100):
            n = random.randint(1, 100)
            m = random.randint(1, n)
            p = random.uniform(0, 0.3)
            f = lambda l, r: l // n != r // n
            trigger = ScheduledStepsTrigger(0, lambda x: x+1, f)

            last_called_step = None

            for step in range(n):
                if random.random() < p:
                    if last_called_step is not None:
                        triggered_ = f(last_called_step, step)
                    else:
                        triggered_ = False

                    last_called_step = step

                    triggered = trigger(step)

                    self.assertEqual(trigger.last_called_step, last_called_step)
                    self.assertEqual(triggered, triggered_)

        for i in range(100):
            n = random.randint(1, 100)
            m = random.randint(1, n)
            p = random.uniform(0, 0.3)
            q = random.uniform(0, 0.3)
            steps = [step for step in range(n) if random.random() < q]
            f = lambda l, r: bisect.bisect_right(steps, l) < \
                             bisect.bisect_right(steps, r)
            trigger = ScheduledStepsTrigger(0, lambda x: x+1, steps)

            last_called_step = -1

            for step in range(n):
                if random.random() < p:
                    triggered_ = f(last_called_step, step)
                    last_called_step = step

                    triggered = trigger(step)

                    self.assertEqual(triggered, triggered_)

        trigger = ScheduledStepsTrigger(0, lambda x: x+1, [])
        for step in range(100):
            trigger.trigger()
            self.assertEqual(trigger.user_state, step+1)


class BestEverConvergenceTriggerTest(tf.test.TestCase):
    """Tests :class:`~texar.utils.BestEverConvergenceTrigger`.
    """

    def test(self):
        for i in range(100):
            n = random.randint(1, 100)
            seq = list(range(n))
            random.shuffle(seq)
            threshold_steps = random.randint(0, n // 2 + 1)
            minimum_interval_steps = random.randint(0, n // 2 + 1)
            trigger = BestEverConvergenceTrigger(
                0, lambda x: x+1, threshold_steps, minimum_interval_steps)

            best_ever_step, best_ever_score, last_triggered_step = -1, -1, None

            for step, score in enumerate(seq):
                if score > best_ever_score:
                    best_ever_step = step
                    best_ever_score = score

                triggered_ = step - best_ever_step >= threshold_steps and \
                    (last_triggered_step is None or
                     step - last_triggered_step >= minimum_interval_steps)
                if triggered_:
                    last_triggered_step = step

                triggered = trigger(step, score)

                self.assertEqual(trigger.best_ever_step, best_ever_step)
                self.assertEqual(trigger.best_ever_score, best_ever_score)
                self.assertEqual(trigger.last_triggered_step,
                                 last_triggered_step)
                self.assertEqual(triggered, triggered_)

        trigger = BestEverConvergenceTrigger(0, lambda x: x+1, 0, 0)
        for step in range(100):
            trigger.trigger()
            self.assertEqual(trigger.user_state, step+1)
 

if __name__ == "__main__":
    tf.test.main()

