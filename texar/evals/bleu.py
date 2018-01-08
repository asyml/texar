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

import tensorflow as tf

# pylint: disable=too-many-locals, no-member, redefined-variable-type

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

    # Create a temporary folder containing hyperthesis and reference files
    result_path = tempfile.mkdtemp()
    # Create hyperthesis file
    hfile_path = os.path.join(result_path, 'hyp')
    with open(hfile_path, "w") as hfile:
        hfile.write("\n".join(hypotheses).encode("utf-8"))
        hfile.write("\n")
    # Create reference files
    max_nrefs = max([len(ref) for ref in references])
    rfile_path = os.path.join(result_path, 'ref')
    for rid in range(max_nrefs):
        with open(rfile_path + '%d'%rid, "w") as rfile:
            for ref in references:
                if rid < len(ref):
                    rfile.write(ref[rid].encode("utf-8") + "\n")
                else:
                    rfile.write("\n")

    # Calculate BLEU
    with open(hfile_path, "r") as hyp_input:
        multi_bleu_cmd = [multi_bleu_path]
        if lowercase:
            multi_bleu_cmd += ["-lc"]
        multi_bleu_cmd += [rfile_path]
        try:
            multi_bleu_ret = subprocess.check_output(
                multi_bleu_cmd, stdin=hyp_input, stderr=subprocess.STDOUT)
            multi_bleu_ret = multi_bleu_ret.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", multi_bleu_ret).group(1)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                tf.logging.warning(
                    "multi-bleu.perl eturned non-zero exit code")
                tf.logging.warning(error.output)
            bleu_score = np.float32(0.0)


    return np.float32(bleu_score)
