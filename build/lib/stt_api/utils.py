from shutil import rmtree
from pathlib import Path
from packaging import version
from types import ModuleType
from itertools import chain
from typing import Any
import importlib
import logging
import os

try:
    import tensorflow as tf
except BaseException:
    tf = None

logger = logging.getLogger(__name__)
DEVICES = None

MALAYA_CACHE = os.environ.get('MALAYA_CACHE', '~/.cache/malaya')


def _delete_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))


def _get_home(package):
    home = os.environ.get(
        f'{package.upper()}_CACHE',
        os.path.join(str(Path.home()), package.title()),
    )
    version_path = os.path.join(home, 'version')
    return home, version_path


def _delete_macos(package):
    home, _ = _get_home(package)
    macos = os.path.join(home, '__MACOSX')
    if os.path.exists(macos):
        rmtree(macos)


def get_cache_dir(path):
    cache_path = os.path.expanduser(MALAYA_CACHE)
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    return os.path.join(cache_path, path)


def get_home(package, package_version):
    home, version_path = _get_home(package)

    try:
        if not os.path.exists(home):
            os.makedirs(home)
    except BaseException:
        raise Exception(
            f'Malaya cannot make directory for caching. Please check your {home}'
        )

    _delete_macos(package=package)
    if not os.path.isfile(version_path):
        with open(version_path, 'w') as fopen:
            fopen.write(package_version)
    else:
        with open(version_path, 'r') as fopen:
            cached_version = fopen.read()
        try:
            if float(cached_version) < 1:
                _delete_folder(home)
                with open(version_path, 'w') as fopen:
                    fopen.write(package_version)
        except BaseException:
            with open(version_path, 'w') as fopen:
                fopen.write(package_version)

    return home, version_path


def describe_availability(dict, transpose=True, text=''):
    if len(text):
        logger.info(text)
    try:
        import pandas as pd

        df = pd.DataFrame(dict)

        if transpose:
            return df.T
        else:
            return df
    except BaseException:
        return dict


def available_device(refresh=False):
    """
    Get list of devices and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """
    global DEVICES

    if DEVICES is None and not refresh:
        from tensorflow.python.client import device_lib

        DEVICES = device_lib.list_local_devices()
        DEVICES = [
            (
                i.name.replace('/device:', ''),
                f'{round(i.memory_limit / 1e9, 3)} GB',
            )
            for i in DEVICES
        ]

    return DEVICES


def available_gpu(refresh=False):
    """
    Get list of GPUs and memory limit from `tensorflow.python.client.device_lib.list_local_devices()`.

    Returns
    -------
    result : List[str]
    """

    devices = available_device(refresh=refresh)
    return [d for d in devices if 'GPU' in d[0] and 'XLA' not in d[0]]


def print_cache(package, location=None):
    """
    Print cached data, this will print entire cache folder if let location = None.

    Parameters
    ----------
    location : str, (default=None)
        if location is None, will print entire cache directory.

    """

    home, _ = _get_home(package=package)
    path = os.path.join(home, location) if location else home
    paths = DisplayablePath.make_tree(Path(path))
    for path in paths:
        print(path.displayable())


def delete_cache(package, location):
    """
    Remove selected cached data, please run print_cache() to get path.

    Parameters
    ----------
    location : str

    Returns
    -------
    result : boolean
    """

    home, _ = _get_home(package=package)
    if not isinstance(location, str):
        raise ValueError('location must be a string')
    location = os.path.join(home, location)
    if not os.path.exists(location):
        raise Exception(
            f"folder not exist, please check path from `{package.replace('-', '_')}.utils.print_cache()`"
        )
    if not os.path.isdir(location):
        raise Exception(
            f"Please use parent directory, please check path from `{package.replace('-', '_')}.utils.print_cache()`"
        )
    _delete_folder(location)
    return True


def delete_all_cache(package):
    """
    Remove cached data, this will delete entire cache folder.
    """
    _delete_macos(package)
    home, _ = _get_home(package)
    try:
        _delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)
        return True
    except BaseException:
        raise Exception(
            f'failed to clear cached models. Please make sure {home} is able to overwrite from {package}'
        )


def close_session(model):
    """
    Close session from a model to prevent any out-of-memory or segmentation fault issues.

    Parameters
    ----------
    model : Malaya object.

    Returns
    -------
    result : boolean
    """

    success = False
    try:
        if hasattr(model, 'sess'):
            model.sess.close()
            success = True
        elif hasattr(model, '_sess'):
            model._sess.close()
            success = True
    except Exception as e:
        logger.warning(e)
    return success


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + os.path.sep
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria
        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path,
                    parent=displayable_root,
                    is_last=is_last,
                    criteria=criteria,
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + os.path.sep
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ['{!s} {!s}'.format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return ''.join(reversed(parts))


def check_tf(func):
    def inner1(*args, **kwargs):

        if tf is None:
            raise ModuleNotFoundError(
                'tensorflow not installed. Please install it by `pip install tensorflow` and try again.'
            )
        return func(*args, **kwargs)

    return inner1


@check_tf
def check_tf2_huggingface():
    if version.parse(tf.__version__) < version.parse('2.0'):
        raise Exception('Tensorflow version must >= 2.0 to use HuggingFace models.')


def check_tf2(func):

    def inner1(*args, **kwargs):

        check_tf2_huggingface()
        return func(*args, **kwargs)

    return inner1


def is_tf(func):

    def inner1(*args, **kwargs):
        logger.info(
            'this interface is a Tensorflow model, make sure you read https://www.tensorflow.org/guide/gpu on how to use hardware accelerator')
        return func(*args, **kwargs)

    return inner1


def is_pytorch(func):

    def inner1(*args, **kwargs):
        logger.info(
            'this interface is a PyTorch model, make sure you read https://pytorch.org/docs/stable/notes/cuda.html on how to use hardware accelerator')
        return func(*args, **kwargs)

    return inner1


def get_module(path, library='malaya'):
    module = path.split(f'{library}/')[-1]
    module = module.replace(os.path.sep, '.').replace('.py', '')
    return f'{library}.{module}'


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of
        # self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}") from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
