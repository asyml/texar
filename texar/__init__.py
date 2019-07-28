# Copyright 2019 The Texar Authors. All Rights Reserved.
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
"""
Modules of texar library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import

import sys

if sys.version_info.major < 3:
    # PY 2.x, import as is because Texar-PyTorch cannot be installed.
    from texar.version import VERSION as __version__

    from texar.module_base import *
    from texar.hyperparams import *
    from texar.context import *
    from texar import modules
    from texar import core
    from texar import losses
    from texar import models
    from texar import data
    from texar import evals
    from texar import agents
    from texar import run
    from texar import utils
else:
    # Lazily load Texar-TF modules upon usage. This is to ensure that Texar-TF
    # and TensorFlow will not be imported if the user only requires
    # Texar-PyTorch modules from `texar.torch`.
    #
    # Due to the lazy loading mechanism, it is now impossible to write
    # `from texar import <module>` within library code (i.e., code that will be
    # accessible from the `texar` module). Please use the following workarounds
    # instead:
    #
    # 1. To import a class / function that is directly accessible from `texar`,
    #    import them from their containing modules. For instance:
    #
    #    `from texar import HParams`
    #      ->  `from texar.hyperparams import HParams`
    #    `from texar import ModuleBase`
    #      ->  `from texar.module_base import ModuleBase`
    # 2. To import a module that is directly accessible from `texar`, use the
    #    `import ... as` syntax. For instance:
    #
    #    `from texar import utils`  ->  `import texar.utils as utils`
    #    `from texar import context`  ->  `import texar.context as context`

    import importlib

    __import_modules__ = [
        "modules", "core", "losses", "models", "data", "evals",
        "agents", "run", "utils",
    ]
    __import_star_modules__ = ["module_base", "hyperparams", "context"]


    def _import_all():
        from texar.version import VERSION
        globals()["__version__"] = VERSION

        for module_name in __import_star_modules__:
            # from ... import *. Requires manually handling `__all__`.
            module = importlib.import_module("." + module_name, package="texar")
            try:
                variables = module.__all__
            except AttributeError:
                variables = [name for name in module.__dict__
                             if not name.startswith("_")]
            globals().update({
                name: module.__dict__[name] for name in variables})

        for module_name in __import_modules__:
            # from ... import module
            module = importlib.import_module("." + module_name, package="texar")
            globals()[module_name] = module


    class _DummyTexarBaseModule:
        # Credit: https://stackoverflow.com/a/7668273/4909228
        def __getattr__(self, name):
            if name in globals():
                # Shortcut to global names.
                return globals()[name]
            if name == "torch":
                # To use `texar.torch`, Texar-TF and TensorFlow should not be
                # imported.
                module = importlib.import_module(".torch", package="texar")
                globals()["torch"] = module
                return module

            # The user tries to access Texar-TF modules, so we load all modules
            # at this point, and restore the registered `texar` module.
            sys.modules[__name__] = __module__
            _import_all()
            return globals()[name]


    # Save `texar` module as `__module__`, ans replace the system-wide
    # registered module with our dummy module.
    __module__ = sys.modules[__name__]
    sys.modules[__name__] = _DummyTexarBaseModule()
