"""Common utils.."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import time
import random
import numpy as np

def log_print(line):
  """Add time to print function."""
  print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
        + "] " + line)

