# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""ProtMyth Module Mapper

This module provides utilities for mapping between target and source modules in PyTorch models.
It includes functionality for transforming and aligning tensors between different state dictionaries,
which can be particularly useful for model conversion, weight transfer, or fine-tuning tasks.

Main Components:
- _TensorTransformFn: A type alias for functions that transform tensors.
- _TensorDictTransformFn: A type alias for functions that transform dictionaries of tensors.
- ModuleMapper: A class that manages the mapping between source and target modules, allowing for
  complex transformations and submodule handling.

Utilities:
- _noops: A no-operation function that returns the input tensor unchanged.
- _subdict_with_prefix: Extracts a subdictionary from a dictionary based on a key prefix.
- _concat_prefix: Concatenates prefixes and names, ensuring proper formatting.
- _append_dot: Appends a dot to a string if it's not empty.
- create_target_state_dict: Clones a model's state dictionary for use as a target in mapping operations.

Dataclasses:
- _MappingEntry: Represents a mapping entry with a source name and a transformation function.
- _MultiMappingEntry: Represents a multi-mapping entry with multiple source names and a transformation function.
- _MappingResult: Stores the result of a mapping operation, including matched source and target names.

Example Usage:
To use the `ModuleMapper`, create an instance and add mapping entries using the `add` or `add_multimap` methods.
Then, call the instance with target and source state dictionaries to execute the mapping process.
"""


from jaxtyping import Float, jaxtyped
from einops import rearrange
import torch
import dataclasses
from typing import Callable, Dict, List, Union

_TensorTransformFn = Callable[
    [Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]],
    Float[torch.Tensor, "..."]
]
_TensorDictTransformFn = Callable[
    [Dict[str, Float[torch.Tensor, "..."]], Float[torch.Tensor, "..."]],
    Float[torch.Tensor, "..."]
]


@jaxtyped
def _noops(x: Float[torch.Tensor, "..."], _: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    """No-operation function that returns the input tensor unchanged.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    _ : torch.Tensor
        An unused tensor parameter.

    Returns
    -------
    torch.Tensor
        The unchanged input tensor.
    """
    return x


def _subdict_with_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extracts a subdictionary with keys that start with the given prefix.

    Parameters
    ----------
    d : Dict[str, torch.Tensor]
        The original dictionary from which to extract the subdictionary.
    prefix : str
        The prefix to filter the keys.

    Returns
    -------
    Dict[str, torch.Tensor]
        A subdictionary containing only the keys that start with the prefix.
    """
    prefix = prefix.strip(".")
    if prefix:
        prefix += "."
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def _concat_prefix(prefix: str, name: str) -> str:
    """Concatenates a prefix and a name, ensuring proper formatting.

    Parameters
    ----------
    prefix : str
        The prefix to concatenate.
    name : str
        The name to concatenate with the prefix.

    Returns
    -------
    str
        The concatenated string.
    """
    prefix = prefix.strip(".")
    name = name.strip(".")
    return f"{prefix}.{name}" if prefix and name else f"{prefix}{name}"


@dataclasses.dataclass
class _MappingEntry:
    """Represents a mapping entry with a source name and a transformation function.

    Attributes
    ----------
    source_name : str
        The name of the source tensor.
    transform : _TensorTransformFn
        The transformation function to apply to the source tensor.
    """
    source_name: str
    transform: _TensorTransformFn = _noops


@dataclasses.dataclass
class _MultiMappingEntry:
    """Represents a multi-mapping entry with multiple source names and a transformation function.

    Attributes
    ----------
    source_names : List[str]
        A list of source tensor names.
    transform : _TensorDictTransformFn
        The transformation function to apply to the source tensors.
    """
    source_names: List[str]
    transform: _TensorDictTransformFn

    @property
    def source_name(self) -> str:
        """Returns a string representation of the source names.

        Returns
        -------
        str
            A string containing all source names.
        """
        return "(" + ", ".join(self.source_names) + ")"


@dataclasses.dataclass
class _MappingResult:
    """Stores the result of a mapping operation, including matched source and target names.

    Attributes
    ----------
    matched_source : set[str]
        A set of matched source names.
    matched_target : set[str]
        A set of matched target names.
    """
    matched_source: set[str] = dataclasses.field(default_factory=set)
    matched_target: set[str] = dataclasses.field(default_factory=set)


class ModuleMapper:
    """A class for mapping between target and source modules.

    Attributes
    ----------
    target_prefix : str
        The prefix for target keys.
    source_prefix : str
        The prefix for source keys.
    entries : Dict[str, Union[_MappingEntry, _MultiMappingEntry]]
        A dictionary of mapping entries.
    submodules : Dict[str, ModuleMapper]
        A dictionary of submodules.
    parent : Union[ModuleMapper, None]
        A reference to the parent module mapper.
    """

    def __init__(self, target_prefix: str = "", source_prefix: str = "") -> None:
        """Initializes the ModuleMapper with target and source prefixes.

        Parameters
        ----------
        target_prefix : str, optional
            The prefix for target keys (default is an empty string).
        source_prefix : str, optional
            The prefix for source keys (default is an empty string).
        """
        super().__init__()
        self.target_prefix = target_prefix
        self.source_prefix = source_prefix
        self.entries: Dict[str, Union[_MappingEntry, _MultiMappingEntry]] = {}
        self.submodules: Dict[str, ModuleMapper] = {}
        self.parent: Union[ModuleMapper, None] = None

    def add(self, target_key: str, source_key: str, transform: _TensorTransformFn = _noops) -> "ModuleMapper":
        """Adds a mapping entry to the mapper.

        Parameters
        ----------
        target_key : str
            The target key for the mapping.
        source_key : str
            The source key for the mapping.
        transform : _TensorTransformFn, optional
            The transformation function to apply (default is no-op).

        Returns
        -------
        ModuleMapper
            The current instance of ModuleMapper.
        """
        self.entries[target_key] = _MappingEntry(source_key, transform)
        return self

    def add_multimap(self, target_key: str, source_keys: List[str],
                     transform: _TensorDictTransformFn) -> "ModuleMapper":
        """Adds a multi-mapping entry to the mapper.

        Parameters
        ----------
        target_key : str
            The target key for the mapping.
        source_keys : List[str]
            A list of source keys for the mapping.
        transform : _TensorDictTransformFn
            The transformation function to apply.

        Returns
        -------
        ModuleMapper
            The current instance of ModuleMapper.
        """
        self.entries[target_key] = _MultiMappingEntry(source_keys, transform)
        return self

    def add_submodule(self, submodule: "ModuleMapper") -> "ModuleMapper":
        """Adds a submodule to the mapper.

        Parameters
        ----------
        submodule : ModuleMapper
            The submodule to add.

        Returns
        -------
        ModuleMapper
            The current instance of ModuleMapper.
        """
        self.submodules[submodule.target_prefix] = submodule
        submodule.parent = self
        return self

    @property
    def full_source_prefix(self) -> str:
        """Returns the full source prefix, including parent prefixes.

        Returns
        -------
        str
            The full source prefix.
        """
        if self.parent is None:
            return self.source_prefix
        return _concat_prefix(self.parent.full_source_prefix, self.source_prefix)

    @property
    def full_target_prefix(self) -> str:
        """Returns the full target prefix, including parent prefixes.

        Returns
        -------
        str
            The full target prefix.
        """
        if self.parent is None:
            return self.target_prefix
        return _concat_prefix(self.parent.full_target_prefix, self.target_prefix)

    def _apply_mapping(
        self,
        target_name: str,
        source_subdict: Dict[str, torch.Tensor],
        target_subdict: Dict[str, torch.Tensor],
        mapping_result: _MappingResult,
    ) -> None:
        """Applies the mapping from source to target for a given entry.

        Parameters
        ----------
        target_name : str
            The name of the target tensor.
        source_subdict : Dict[str, torch.Tensor]
            The source subdictionary containing tensors.
        target_subdict : Dict[str, torch.Tensor]
            The target subdictionary containing tensors.
        mapping_result : _MappingResult
            The mapping result to update.
        """
        entry = self.entries[target_name]
        target_tensor = target_subdict[target_name]
        full_target_name = _concat_prefix(self.full_target_prefix, target_name)

        if isinstance(entry, _MappingEntry):
            full_source_name = _concat_prefix(self.full_source_prefix, entry.source_name)
            print(f"[{self.__class__.__name__}] Mapping {full_source_name} -> {full_target_name}")
            source_tensor = source_subdict[entry.source_name]

            # Assuming the transformation can be expressed as a rearrangement.
            target_tensor.data.copy_(
                entry.transform(rearrange(source_tensor, '... -> ...'), target_tensor)
            )

            mapping_result.matched_source.add(full_source_name)
            mapping_result.matched_target.add(full_target_name)

        elif isinstance(entry, _MultiMappingEntry):
            full_source_names = [
                _concat_prefix(self.full_source_prefix, source_name) for source_name in entry.source_names
            ]
            print(f"[{self.__class__.__name__}] Mapping ({', '.join(full_source_names)}) -> {full_target_name}")
            source_tensor_dict = {source_name: source_subdict[source_name].data for source_name in entry.source_names}

            target_tensor.data.copy_(
                entry.transform(source_tensor_dict, target_tensor)
            )

            mapping_result.matched_source.update(full_source_names)
            mapping_result.matched_target.add(full_target_name)

    def __call__(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        source_state_dict: Dict[str, torch.Tensor],
        mapping_result: Union[_MappingResult, None] = None,
    ) -> _MappingResult:
        """Executes the mapping process between target and source state dictionaries.

        Parameters
        ----------
        target_state_dict : Dict[str, torch.Tensor]
            The target state dictionary.
        source_state_dict : Dict[str, torch.Tensor]
            The source state dictionary.
        mapping_result : Union[_MappingResult, None], optional
            An optional mapping result to update (default is None).

        Returns
        -------
        _MappingResult
            The result of the mapping operation.
        """
        if mapping_result is None:
            mapping_result = _MappingResult()

        source_subdict = _subdict_with_prefix(source_state_dict, self.full_source_prefix)
        target_subdict = _subdict_with_prefix(target_state_dict, self.full_target_prefix)

        for submodule in self.submodules.values():
            submodule(target_state_dict, source_state_dict, mapping_result=mapping_result)

        for target_name in target_subdict.keys():
            if target_name in self.entries:
                self._apply_mapping(
                    target_name=target_name,
                    source_subdict=source_subdict,
                    target_subdict=target_subdict,
                    mapping_result=mapping_result
                )

        return mapping_result

    def get_repr_str(self, depth: int = 0) -> str:
        """Generates a string representation of the mapper for debugging purposes.

        Parameters
        ----------
        depth : int, optional
            The current depth for indentation (default is 0).

        Returns
        -------
        str
            A string representation of the mapper.
        """
        indent = "  " * depth
        rows = [indent + "{{" + self.__class__.__name__ if not depth else '']

        for target_name, entry in self.entries.items():
            target_full = _concat_prefix(self.full_target_prefix, target_name)
            source_full = _concat_prefix(self.full_source_prefix, entry.source_name)
            rows.append(f"{indent}  {target_full} <- {source_full} ({entry.transform.__name__}), ")

        for submodule in self.submodules.values():
            rows.append(submodule.get_repr_str(depth + 1))

        rows.append(indent + "}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        """Returns the string representation of the mapper.

        Returns
        -------
        str
            The string representation of the mapper.
        """
        return self.get_repr_str()


@jaxtyped
def create_target_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Creates a target state dictionary by cloning the model's state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to create the target state dictionary.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing the cloned state of the model.
    """
    return {k: v.clone() for k, v in model.state_dict().items()}
