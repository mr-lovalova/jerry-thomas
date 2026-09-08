import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _UniqueKeyLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        initial: [
            (tag, pattern)
            for tag, pattern in resolvers
            if tag != "tag:yaml.org,2002:bool"
        ]
        for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        keys: set[Any] = set()
        has_merge_key = False
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                if has_merge_key:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found duplicate key '<<'",
                        key_node.start_mark,
                    )
                has_merge_key = True
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in keys
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


@dataclass(frozen=True, slots=True)
class YamlDocument:
    path: Path
    data: Any


def read_yaml_document(path: Path, require_mapping: bool = True) -> YamlDocument:
    try:
        content = path.read_bytes()
        data = yaml.load(content.decode("utf-8"), Loader=_UniqueKeyLoader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAML file not found: {path}") from exc
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        data = {}
    if require_mapping and not isinstance(data, dict):
        raise TypeError(
            f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}"
        )
    return YamlDocument(path=path.resolve(), data=data)


def load_yaml(path: Path, require_mapping: bool = True) -> Any:
    return read_yaml_document(path, require_mapping=require_mapping).data
