from typing import Annotated

from pydantic import StringConstraints


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

DottedIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$",
    ),
]
