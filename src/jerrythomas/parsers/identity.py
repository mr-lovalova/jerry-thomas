from typing import Any

from jerrythomas.sources.parser import DataParser


class IdentityParser(DataParser[Any]):
    """Pass-through parser: returns the input unchanged.

    Useful when a loader (or generator) already produces domain records.
    """

    def parse(self, raw: Any) -> Any:
        return raw
