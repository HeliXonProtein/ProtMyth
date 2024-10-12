# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Module registration for module hub.
"""
from typing import Literal, Callable

from protmyth.modules.base import BaseModule

_MODULES: dict[str, dict[str, BaseModule]] = {
    "auxilary": {},
    "common": {},
    "embeddings": {},
    "losses": {},
    "seqencoders": {},
    "seqdecoders": {},
    "structencoders": {},
    "structdeocders": {},
}


def register_module(
    name: Literal[
        "auxilary",
        "common",
        "embeddings",
        "losses",
        "seqencoders",
        "seqdecoders",
        "structencoders",
        "structdeocders"
    ]
) -> Callable:
    """Decorator to register a module.

    Args:
        name (str): Name of the module. Small case, no spaces and should be unique.

    Returns:
        decoder (Callable): Decorator function.
    """
    def _decorator(cls: BaseModule):
        """Decorator function.

        Args:
            cls (BaseModule)

        Returns:
            cls (BaseModule)
        """
        assert name in _MODULES, f"Module {name} not found in module hub."
        _MODULES[name][cls.__class__.__name__] = cls
        return cls
    return _decorator
