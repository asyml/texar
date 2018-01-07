#
"""
The BLEU metric.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import re
import subprocess
import tempfile
import numpy as np

from six.moves import urllib
import tensorflow as tf


def bleu(hypotheses, references, lowercase=False):
    """Calculates BLEU score using the MOSES multi-bleu.perl script.

    Args:
        hypotheses (list): A list of strings.
            Each string is a single hypothesis example. List can also be numpy
            array.
        references (list of list): A list of reference list for each of
            the hypotheses. Each reference list is a list of string where
            each string is a single reference example. List can also be
            numpy array.
        lowercase (bool): If true, pass the "-lc" flag to the multi-bleu script

    Returns:
        float32: the BLEU score.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.)   # pylint: disable=no-member

    # Get multi-bleu.perl
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    multi_bleu_path = os.path.abspath(
        os.path.join(cur_dir, "..", "..", "tools", "multi-bleu.perl"))

    # Create hyperthesis file
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()

    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
      bleu_cmd = [multi_bleu_path]
      if lowercase:
        bleu_cmd += ["-lc"]
      bleu_cmd += [reference_file.name]
      try:
        bleu_out = subprocess.check_output(
            bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        bleu_score = float(bleu_score)
      except subprocess.CalledProcessError as error:
        if error.output is not None:
          tf.logging.warning("multi-bleu.perl script returned non-zero exit code")
          tf.logging.warning(error.output)
        bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return np.float32(bleu_score)
