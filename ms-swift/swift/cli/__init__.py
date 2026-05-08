"""Command-line argument contracts for the used swift entrypoints."""

from .contract import SwiftCommand, parse_swift_command

__all__ = ["SwiftCommand", "parse_swift_command"]
