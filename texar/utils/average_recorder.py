#
"""
Utilities for maintaining moving average.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque

from texar.utils.exceptions import TexarError

__all__ = [
    "_SingleAverageRecorder",
    "AverageRecorder"
]

class _SingleAverageRecorder(object):
    """Maintains the moving average (i.e., the average of the latest N records)
    of a single metric.

    Args:
        size (int, optional): The window size of moving average. If `None`,
            the average of all added records is maintained.
    """

    def __init__(self, size=None):
        if size is not None and size <= 0:
            raise ValueError("`size` must be > 0 or `None`.")
        self._size = size
        self._q = deque([])
        self._sum = 0.
        self._num = 0

    def add(self, record):
        """Appends a new record.

        Args:
            record: A scalar; the new record to append.

        Returns:
            The (moving) average after appending the record.
        """
        self._sum += record
        self._num += 1

        if self._size is not None:
            if len(self._q) == self._size:
                self._sum -= self._q.popleft()
            self._q.append(record)
            self._num = min(self._num, self._size)

            if len(self._q) != self._num:
                raise TexarError(
                    "Internal error: length of `_q` does not match `_num`.")

        return self.avg()

    def avg(self):
        """Returns the (moving) average.
        """
        if self._num == 0:
            return 0.
        return self._sum / self._num

    def reset(self):
        """Cleans all records.
        """
        self._q.clear()
        self._sum = 0.
        self._num = 0


class AverageRecorder(object):
    """Maintains the moving average (i.e., the average of the latest N records)
    of possibly multiple metrics.

    The metrics are determined by the first call of :meth:`add`.

    Args:
        size (int, optional): The window size of moving average. If `None`,
            the average of all added records is maintained.
    """

    def __init__(self, size=None):
        if size is not None and size <= 0:
            raise ValueError("`size` must be > 0 or `None`.")
        self._size = size
        self._recorders = None
        self._default_metric_name = "metric"
        self._record_type = None

    def _to_dict(self, record):
        if isinstance(record, dict):
            record_dict = record
        elif isinstance(record, (list, tuple)):
            record_dict = {i: vi for i, vi in enumerate(record)}
        else:
            record_dict = {self._default_metric_name: record}
        return record_dict

    def add(self, record):
        """Appends a new record.

        :attr:`record` can be a `list`, `dict`, or a single scalar. The
        record type is determined at the first time :meth:`add` is called.
        All subsequent calls to :meth:`add` must have the same type of
        :attr:`record`.

        :attr:`record` in subsequent calls to :meth:`add` can contain only
        a subset of fields than the first call to :meth:`add`.

        Example:
            .. code-block:: python

                recorder.add({1: v1, 2: v2}) # 1st call to `add`
                x = recorder.add({1: v3}) # 2nd call to `add`
                # x == {1: (v1 + v3) / 2, 2: v2}

        Args:
            record: A single scalar, a list of scalars, or a dict of scalars.

        Returns:
            The (moving) average after appending the record, with the same
            type as :attr:`record`.
        """
        if self._record_type is None:
            self._record_type = type(record)
        elif self._record_type != type(record):
            raise ValueError('The type of `record` is not consistent. '
                             'Expect type `{}`'.format(self._record_type))

        record_dict = self._to_dict(record)
        if self._recorders is None:
            self._recorders = {
                name: _SingleAverageRecorder(self._size)
                for name in record_dict.keys()
            }

        for name, val in record_dict.items():
            self._recorders[name].add(val)

        return self.avg()

    def avg(self, id_or_name=None):
        """Returns the (moving) average.

        Args:
            id_or_name (optional): A list or a single element. Each element is
                the index (if the record type is `list`) or name (if the
                record type is `dict`) of the field to calculate average.
                If `None`, the average of all fields are returned.

        Returns:
            The average, with the same type as record.
        """
        if self._recorders is None:
            return 0.

        keys = id_or_name
        if keys is None:
            keys = list(self._recorders.keys())
        elif not isinstance(keys, (list, tuple)):
            keys = [keys]

        avg = {key: self._recorders[key].avg() for key in keys}
        if self._record_type in {list, tuple}:
            ret_avg = []
            for k, v in avg.items():
                if k in keys:
                    ret_avg.append(v)
            return self._record_type(ret_avg)
        elif self._record_type == dict:
            return avg
        else:
            return avg[self._default_metric_name]

    def reset(self, id_or_name=None):
        """Resets the record.

        id_or_name (optional): A list or a single element. Each element is
            the index (if the record type is `list`) or name (if the
            record type is `dict`) of the field to reset.
            If `None`, all fields are reset.
        """
        keys = id_or_name
        if keys is None:
            keys = list(self._recorders.keys())
        elif not isinstance(keys, (list, tuple)):
            keys = [keys]

        for key in keys:
            self._recorders[key].reset()
