#
"""
Utilities for maintaining moving average.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque

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
        name (str, optional): name of the recorder. Used when printing.
    """

    def __init__(self, size=None, name=None):
        if size is not None and size <= 0:
            raise ValueError("`size` must be > 0 or `None`.")
        self._size = size
        self._q = deque([])
        self._w = deque([])
        self._sum = 0.
        self._w_sum = 0
        self._name = name

    def add(self, record, weight=None):
        """Appends a new record.

        Args:
            record: A scalar; the new record to append.
            weight (optional): A scalar, weight of the new record for
                calculating a weighted average. If `None`, weight is set to `1`.
                For example, :attr:`weight` can be set to batch size and
                :attr:`record` the average value of certain metric on the batch
                in order to calculate the average metric value on a whole
                dataset.

        Returns:
            The (moving) average after appending the record.
        """
        w = weight if weight is not None else 1
        self._w_sum += w
        self._sum += record * w

        if self._size is not None:
            if len(self._q) == self._size:
                w_pop = self._w.popleft()
                self._sum -= self._q.popleft() * w_pop
                self._w_sum -= w_pop
            self._q.append(record)
            self._w.append(w)

        return self.avg()

    def avg(self):
        """Returns the (moving) average.
        """
        if self._w_sum == 0:
            return 0.
        return self._sum / self._w_sum

    def reset(self):
        """Cleans all records.
        """
        self._q.clear()
        self._w.clear()
        self._sum = 0.
        self._w_sum = 0

    def to_str(self, precision=None):
        """Returns a string of the average value.

        Args:
            precision (int, optional): The number of decimal places to keep in
                the returned string. E.g., for an average value of `0.1234`,
                :attr:`precision = 2` leads to `'0.12'`.

        Returns:
            A string of the average value. If :meth:`name` is given, the
            string is of the format like `'name: 0.1234'`, otherwise
            the string is of the format like `'0.1234'`.
        """
        prec_str = "{}"
        if precision is not None:
            prec_str = "{:.%df}" % precision

        avg_str = prec_str.format(self.avg())
        if self._name is not None:
            avg_str = "{}: {}".format(self._name, avg_str)

        return avg_str

    @property
    def name(self):
        """The name of the recorder.
        """
        return self.name

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

    def add(self, record, weight=None):
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
            weight (optional): A scalar, weight of the new record for
                calculating a weighted average. If `None`, weight is set to `1`.
                For example, :attr:`weight` can be set to batch size and
                :attr:`record` the average value of certain metrics on the batch
                in order to calculate the average metric values on a whole
                dataset.

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
                name: _SingleAverageRecorder(
                    self._size, name if self._record_type == dict else None)
                for name in record_dict.keys()
            }

        for name, val in record_dict.items():
            self._recorders[name].add(val, weight=weight)

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

    def to_str(self, precision=None, delimiter=' '):
        """Returns a string of the average values of the records.

        Args:
            precision (int, optional): The number of decimal places to keep in
                the returned string. E.g., for an average value of `0.1234`,
                :attr:`precision = 2` leads to `'0.12'`.
            delimiter (str): The delimiter string that separates between
                fields.

        Returns:
            A string of the average values.

            If record is of type `dict`, the string is a concatenation of
            'field_name: average_value', delimited with :attr:`delimiter`.
            E.g., `'field_name_1: 0.1234 field_name_2: 0.5678 ...'`.

            Otherwise, the string is of a concatenation of 'average_value'.
            E.g., `'0.1234 0.5678 ...'`
        """
        strs = {name: rec.to_str(precision=precision)
                for name, rec in self._recorders.items()}
        str_list = []
        if self._record_type in {list, tuple}:
            for i in range(len(strs)):
                str_list.append(strs[i])
        elif self._record_type == dict:
            str_list = list(strs.values())
        else:
            str_list = [strs[self._default_metric_name]]

        avg_str = delimiter.join(str_list)

        return avg_str
