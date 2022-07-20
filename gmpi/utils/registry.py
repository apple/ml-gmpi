#!/usr/bin/env python3

# modified from https://github.com/facebookresearch/habitat-lab/blob/0e1d2afe04b5856e6b5b3fc561adcea414091be4/habitat/core/registry.py

import collections
from typing import Any, Callable, DefaultDict, Dict, Optional, Type


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: str,
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(to_register, assert_type), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def register_engine(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a training engine to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        # from tex_gen.tex_smoother.engine import BaseEngine
        # return cls._register_impl("engine", to_register, name, assert_type=BaseEngine)
        return cls._register_impl(
            "engine",
            to_register,
            name,
        )

    @classmethod
    def get_engine(cls, name):
        return cls._get_impl("engine", name)

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a Visual Odometry model to registry with key 'name'.

        Args:
            name: Key with which the VO model will be registered.
                If None will use the name of the class.

        """
        return cls._register_impl(
            "model",
            to_register,
            name,
        )  # assert_type=VO

    @classmethod
    def get_model(cls, name):
        return cls._get_impl("model", name)


registry = Registry()
