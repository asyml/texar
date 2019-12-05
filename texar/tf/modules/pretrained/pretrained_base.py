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
Base class for Pre-trained Modules.
"""
import os
import sys

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from texar.tf.data.data_utils import maybe_download
from texar.tf.hyperparams import HParams
from texar.tf.utils.types import MaybeList


__all__ = [
    "default_download_dir",
    "set_default_download_dir",
    "PretrainedMixin",
]

_default_texar_download_dir = None


def default_download_dir(name: str) -> Path:
    r"""Return the directory to which packages will be downloaded by default.
    """
    global _default_texar_download_dir  # pylint: disable=global-statement
    if _default_texar_download_dir is None:
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            # On Windows, use %APPDATA%
            home_dir = Path(os.environ['APPDATA'])
        else:
            # Otherwise, install in the user's home directory.
            home_dir = Path.home()

        if os.access(home_dir, os.W_OK):
            _default_texar_download_dir = home_dir / 'texar_data'
        else:
            raise ValueError(f"The path {home_dir} is not writable. Please "
                             f"manually specify the download directory")

    if not _default_texar_download_dir.exists():
        _default_texar_download_dir.mkdir(parents=True)

    return _default_texar_download_dir / name


def set_default_download_dir(path):
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValueError("`path` must be a string or a pathlib.Path object")

    if not os.access(path, os.W_OK):
        raise ValueError(
            f"The specified download directory {path} is not writable")

    global _default_texar_download_dir  # pylint: disable=global-statement
    _default_texar_download_dir = path


class PretrainedMixin(ABC):
    r"""A mixin class for all pre-trained classes to inherit.
    """

    _MODEL_NAME: str
    _MODEL2URL: Dict[str, MaybeList[str]]

    pretrained_model_dir: Optional[str]

    @classmethod
    def available_checkpoints(cls):
        return list(cls._MODEL2URL.keys())

    def _name_to_variable(self, name):
        r"""Find the corresponding variable given the specified name.
        """
        pointer = self
        for m_name in name.split("."):
            if m_name.isdigit():
                num = int(m_name)
                pointer = pointer[num]  # type: ignore
            else:
                pointer = getattr(pointer, m_name)
        return pointer  # type: ignore

    def load_pretrained_config(self,
                               pretrained_model_name=None,
                               cache_dir=None,
                               hparams=None):
        r"""Load paths and configurations of the pre-trained model.

        Args:
            pretrained_model_name (optional): A str with the name
                of a pre-trained model to load. If `None`, will use the model
                name in :attr:`hparams`.
            cache_dir (optional): The path to a folder in which the
                pre-trained models will be cached. If `None` (default),
                a default directory will be used.
            hparams (dict or HParams, optional): Hyperparameters. Missing
                hyperparameter will be set to default values. See
                :meth:`default_hparams` for the hyperparameter structure
                and default values.
        """
        if not hasattr(self, "_hparams"):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` is already assigned, but `hparams` "
                    "argument is not None.")

        self.pretrained_model_dir = None
        self.pretrained_model_name = pretrained_model_name

        if self.pretrained_model_name is None:
            self.pretrained_model_name = self._hparams.pretrained_model_name
        if self.pretrained_model_name is not None:
            self.pretrained_model_dir = self.download_checkpoint(
                self.pretrained_model_name, cache_dir)
            pretrained_model_hparams = self._transform_config(
                self.pretrained_model_name, self.pretrained_model_dir)
            self._hparams = HParams(
                pretrained_model_hparams, self._hparams.todict())

    def init_pretrained_weights(self, *args, **kwargs):
        if self.pretrained_model_dir:
            self._init_from_checkpoint(
                self.pretrained_model_name,
                self.pretrained_model_dir, *args, **kwargs)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize parameters of the pre-trained model. This method is only
        called if pre-trained checkpoints are not loaded.
        """
        pass

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "pretrained_model_name": None,
                "name": "pretrained_base"
            }
        """
        return {
            'pretrained_model_name': None,
            'name': "pretrained_base",
            '@no_typecheck': ['pretrained_model_name']
        }

    @classmethod
    def download_checkpoint(cls, pretrained_model_name, cache_dir=None):
        r"""Download the specified pre-trained checkpoint, and return the
        directory in which the checkpoint is cached.

        Args:
            pretrained_model_name (str): Name of the model checkpoint.
            cache_dir (str, optional): Path to the cache directory. If `None`,
                uses the default directory (user's home directory).

        Returns:
            Path to the cache directory.
        """
        if pretrained_model_name in cls._MODEL2URL:
            download_path = cls._MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                f"Pre-trained model not found: {pretrained_model_name}")

        if cache_dir is None:
            cache_path = default_download_dir(cls._MODEL_NAME)
        else:
            cache_path = Path(cache_dir)
        cache_path = cache_path / pretrained_model_name

        if not cache_path.exists():
            if isinstance(download_path, str):
                filename = download_path.split('/')[-1]
                maybe_download(download_path, cache_path, extract=True)
                folder = None
                for file in cache_path.iterdir():
                    if file.is_dir():
                        folder = file
                assert folder is not None
                (cache_path / filename).unlink()
                for file in folder.iterdir():
                    file.rename(file.parents[1] / file.name)
                folder.rmdir()
            else:
                for path in download_path:
                    maybe_download(path, cache_path)
            print(f"Pre-trained {cls._MODEL_NAME} checkpoint "
                  f"{pretrained_model_name} cached to {cache_path}")
        else:
            print(f"Using cached pre-trained {cls._MODEL_NAME} checkpoint "
                  f"from {cache_path}.")

        return str(cache_path)

    @classmethod
    @abstractmethod
    def _transform_config(cls, pretrained_model_name, cache_dir):
        r"""Load the official configuration file and transform it into
        Texar-style hyperparameters.

        Args:
            pretrained_model_name (str): Name of the pre-trained model.
            cache_dir (str): Path to the cache directory.

        Returns:
            dict: Texar module hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_from_checkpoint(self, pretrained_model_name, cache_dir, **kwargs):
        r"""Initialize model parameters from weights stored in the pre-trained
        checkpoint.

        Args:
            pretrained_model_name (str): Name of the pre-trained model.
            cache_dir (str): Path to the cache directory.
            **kwargs: Additional arguments for specific models.
        """
        raise NotImplementedError
