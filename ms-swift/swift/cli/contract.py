"""Tiny parser for the swift command contract used by Train/.

The official CLI maps arguments to dataclasses and launches training. This
reference parser only records flags in a structured object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SwiftCommand:
    entrypoint: str
    args: dict[str, Any]


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_swift_command(entrypoint: str, argv: list[str]) -> SwiftCommand:
    """Parse ``--key value`` pairs while preserving repeated/list arguments."""

    parsed: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"expected --flag, got {token!r}")
        key = token[2:].replace("-", "_")
        index += 1

        values: list[Any] = []
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(_coerce(argv[index]))
            index += 1
        value: Any
        if not values:
            value = True
        elif len(values) == 1:
            value = values[0]
        else:
            value = values

        if key in parsed:
            previous = parsed[key]
            if not isinstance(previous, list):
                previous = [previous]
            parsed[key] = previous + (value if isinstance(value, list) else [value])
        else:
            parsed[key] = value

    return SwiftCommand(entrypoint=entrypoint, args=parsed)


def main(entrypoint: str, argv: list[str]) -> None:
    command = parse_swift_command(entrypoint, argv)
    print(json.dumps({"entrypoint": command.entrypoint, "args": command.args}, indent=2, sort_keys=True))
