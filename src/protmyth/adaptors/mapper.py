# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

import dataclasses
from collections.abc import Callable
import torch

# Type aliases for tensor transformation functions
_TensorTransformFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_TensorDictTransformFn = Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor]


def _noops(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
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


def _subdict_with_prefix(d: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Extracts a subdictionary with keys that start with the given prefix.

    Parameters
    ----------
    d : dict[str, torch.Tensor]
        The original dictionary from which to extract the subdictionary.
    prefix : str
        The prefix to filter the keys.

    Returns
    -------
    dict[str, torch.Tensor]
        A subdictionary containing only the keys that start with the prefix.
    """
    prefix = prefix.strip(".")
    if prefix != "":
        prefix += "."
    subdict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            subdict[k[len(prefix):]] = v
    return subdict


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
    if not prefix or not name:
        return prefix + name
    return prefix + "." + name


def _append_dot(name: str) -> str:
    """Appends a dot to the name if it's not empty.

    Parameters
    ----------
    name : str
        The name to which a dot may be appended.

    Returns
    -------
    str
        The name with a dot appended if it was not empty.
    """
    name = name.strip(".")
    if not name:
        return name
    return name + "."


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
    source_names : list[str]
        A list of source tensor names.
    transform : _TensorDictTransformFn
        The transformation function to apply to the source tensors.
    """
    source_names: list[str]
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
    _entries : dict[str, _MappingEntry | _MultiMappingEntry]
        A dictionary of mapping entries.
    _submodules : dict[str, ModuleMapper]
        A dictionary of submodules.
    _parent : ModuleMapper | None
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
        self._target_prefix = target_prefix
        self._source_prefix = source_prefix
        self._entries: dict[str, _MappingEntry | _MultiMappingEntry] = {}
        self._submodules: dict[str, ModuleMapper] = {}
        self._parent: ModuleMapper | None = None

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
        self._entries[target_key] = _MappingEntry(source_key, transform)
        return self

    def add_multimap(
        self, target_key: str, source_keys: list[str], transform: _TensorDictTransformFn
    ) -> "ModuleMapper":
        """Adds a multi-mapping entry to the mapper.

        Parameters
        ----------
        target_key : str
            The target key for the mapping.
        source_keys : list[str]
            A list of source keys for the mapping.
        transform : _TensorDictTransformFn
            The transformation function to apply.

        Returns
        -------
        ModuleMapper
            The current instance of ModuleMapper.
        """
        self._entries[target_key] = _MultiMappingEntry(source_keys, transform)
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
        self._submodules[submodule._target_prefix] = submodule
        submodule._parent = self
        return self

    @property
    def full_source_prefix(self) -> str:
        """Returns the full source prefix, including parent prefixes.

        Returns
        -------
        str
            The full source prefix.
        """
        if self._parent is None:
            return self._source_prefix
        return _concat_prefix(self._parent.full_source_prefix, self._source_prefix)

    @property
    def full_target_prefix(self) -> str:
        """Returns the full target prefix, including parent prefixes.

        Returns
        -------
        str
            The full target prefix.
        """
        if self._parent is None:
            return self._target_prefix
        return _concat_prefix(self._parent.full_target_prefix, self._target_prefix)

    def _apply_mapping(
        self,
        target_name: str,
        source_subdict: dict[str, torch.Tensor],
        target_subdict: dict[str, torch.Tensor],
        _mapping_result: _MappingResult,
    ) -> None:
        """Applies the mapping from source to target for a given entry.

        Parameters
        ----------
        target_name : str
            The name of the target tensor.
        source_subdict : dict[str, torch.Tensor]
            The source subdictionary containing tensors.
        target_subdict : dict[str, torch.Tensor]
            The target subdictionary containing tensors.
        _mapping_result : _MappingResult
            The mapping result to update.
        """
        entry = self._entries[target_name]
        target_tensor = target_subdict[target_name]
        full_target_name = _concat_prefix(self.full_target_prefix, target_name)
        if isinstance(entry, _MappingEntry):
            full_source_name = _concat_prefix(self.full_source_prefix, entry.source_name)
            print(f"[{self.__class__.__name__}] Mapping {full_source_name} -> {full_target_name}")
            source_tensor = source_subdict[entry.source_name]
            target_tensor.data.copy_(entry.transform(source_tensor.data, target_tensor))
            _mapping_result.matched_source.add(full_source_name)
            _mapping_result.matched_target.add(full_target_name)
        elif isinstance(entry, _MultiMappingEntry):
            full_source_names = [
                _concat_prefix(self.full_source_prefix, source_name) for source_name in entry.source_names
            ]
            print(f"[{self.__class__.__name__}] Mapping ({', '.join(full_source_names)}) -> {full_target_name}")
            source_tensor_dict = {source_name: source_subdict[source_name].data for source_name in entry.source_names}
            target_tensor.data.copy_(entry.transform(source_tensor_dict, target_tensor))
            _mapping_result.matched_source.update(full_source_names)
            _mapping_result.matched_target.add(full_target_name)

    def __call__(
        self,
        target_state_dict: dict[str, torch.Tensor],
        source_state_dict: dict[str, torch.Tensor],
        _mapping_result: _MappingResult | None = None,
    ) -> _MappingResult:
        """Executes the mapping process between target and source state dictionaries.

        Parameters
        ----------
        target_state_dict : dict[str, torch.Tensor]
            The target state dictionary.
        source_state_dict : dict[str, torch.Tensor]
            The source state dictionary.
        _mapping_result : _MappingResult | None, optional
            An optional mapping result to update (default is None).

        Returns
        -------
        _MappingResult
            The result of the mapping operation.
        """
        if _mapping_result is None:
            _mapping_result = _MappingResult()
        source_subdict = _subdict_with_prefix(source_state_dict, self.full_source_prefix)
        target_subdict = _subdict_with_prefix(target_state_dict, self.full_target_prefix)

        for target_prefix in self._submodules.keys():
            self._submodules[target_prefix](target_state_dict,source_state_dict,_mapping_result=_mapping_result)

        for target_name in target_subdict.keys():
            if target_name in self._entries:
                self._apply_mapping(
                    target_name=target_name,
                    source_subdict=source_subdict,
                    target_subdict=target_subdict,
                    _mapping_result=_mapping_result,
                )

        return _mapping_result

    def _get_repr_str(self, depth: int = 0) -> str:
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
        rows = []
        rows.append(indent + f"{{{self.__class__.__name__ if depth == 0 else ''}")
        for target_name, entry in self._entries.items():
            target_full = _concat_prefix(self.full_target_prefix, target_name)
            source_full = _concat_prefix(self.full_source_prefix, entry.source_name)
            rows.append(indent + f"  {target_full} <- {source_full} ({entry.transform.__name__}), ")
        for submodule in self._submodules.values():
            rows.append(submodule._get_repr_str(depth + 1))
        rows.append(indent + "}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        """Returns the string representation of the mapper.

        Returns
        -------
        str
            The string representation of the mapper.
        """
        return self._get_repr_str()


def create_target_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Creates a target state dictionary by cloning the model's state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to create the target state dictionary.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the cloned state of the model.
    """
    return {k: v.clone() for k, v in model.state_dict().items()}