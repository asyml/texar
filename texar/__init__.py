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

# pylint: disable=wildcard-import

import sys

if sys.version_info.major < 3:
    # PY 2.x, import as is because Texar-PyTorch cannot be installed.
    import texar.tf

else:
    # Lazily load Texar-TF modules upon usage. This is to ensure that Texar-TF
    # and TensorFlow will not be imported if the user only requires
    # Texar-PyTorch modules from `texar.torch`.

    import importlib

    __import_modules__ = [
        "modules", "core", "losses", "models", "data", "evals",
        "agents", "run", "utils",
    ]
    __import_star_modules__ = ["module_base", "hyperparams", "context"]

    def _import_all():
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "Importing from `texar` is deprecated. Please import from "
                "`texar.tf` instead, e.g. `import texar.tf as tx`",
                DeprecationWarning, stacklevel=3)

        from texar.tf.version import VERSION
        globals()["__version__"] = VERSION

        for module_name in __import_star_modules__:
            # from ... import *. Requires manually handling `__all__`.
            module = importlib.import_module("." + module_name, package="texar.tf")
            try:
                variables = module.__all__
            except AttributeError:
                variables = [name for name in module.__dict__
                             if not name.startswith("_")]
            globals().update({
                name: module.__dict__[name] for name in variables})
            globals()[module_name] = module

        for module_name in __import_modules__:
            # from ... import module
            module = importlib.import_module("." + module_name, package="texar.tf")
            globals()[module_name] = module

    class _DummyTexarBaseModule:
        # Credit: https://stackoverflow.com/a/7668273/4909228
        def __getattr__(self, name):
            if name in globals():
                # Shortcut to global names.
                return globals()[name]
            if name in ["torch", "tf"]:
                # To use `texar.torch`, Texar-TF and TensorFlow should not be
                # imported; To use `texar.tf`, Texar-PyTorch and PyTorch should
                # not be imported.
                module = importlib.import_module("." + name, package="texar")
                globals()[name] = module
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
