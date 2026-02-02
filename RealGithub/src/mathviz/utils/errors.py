"""
Error types and source location tracking for MathViz compiler.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """
    Represents a location in the source code.

    Attributes:
        line: 1-indexed line number
        column: 1-indexed column number
        offset: 0-indexed byte offset from start of source
        filename: Optional filename for error reporting
    """

    line: int
    column: int
    offset: int = 0
    filename: Optional[str] = None

    def __str__(self) -> str:
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"


class MathVizError(Exception):
    """Base exception for all MathViz compiler errors."""

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        source_line: Optional[str] = None,
    ) -> None:
        self.message = message
        self.location = location
        self.source_line = source_line
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []

        if self.location:
            parts.append(f"[{self.location}]")

        parts.append(self.message)

        if self.source_line and self.location:
            parts.append(f"\n    {self.source_line}")
            # Add caret pointing to the error column
            padding = " " * (4 + self.location.column - 1)
            parts.append(f"\n{padding}^")

        return " ".join(parts) if not self.source_line else parts[0] + " " + "".join(parts[1:])


class LexerError(MathVizError):
    """Raised when the lexer encounters an invalid token or character."""

    pass


class ParserError(MathVizError):
    """Raised when the parser encounters a syntax error."""

    pass


class CodeGenError(MathVizError):
    """Raised when code generation fails."""

    pass


class TypeError(MathVizError):
    """Raised when type checking fails."""

    pass


class ModuleResolutionError(MathVizError):
    """
    Raised when a module cannot be resolved or loaded.

    This error is raised when:
    - A module file cannot be found
    - A circular dependency is detected
    - A private symbol is accessed from outside its module
    - A module path is invalid
    """

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        module_path: Optional[str] = None,
        search_paths: Optional[list] = None,
        cycle: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the error.

        Args:
            message: Error message
            location: Source location where the error occurred
            module_path: The module path that couldn't be resolved
            search_paths: List of paths that were searched
            cycle: The circular dependency chain if applicable
        """
        self.module_path = module_path
        self.search_paths = search_paths or []
        self.cycle = cycle
        super().__init__(message, location)

    def _format_message(self) -> str:
        parts = []

        if self.location:
            parts.append(f"[{self.location}]")

        parts.append(self.message)

        if self.cycle:
            cycle_str = " -> ".join(self.cycle)
            parts.append(f"\n  Cycle: {cycle_str}")

        if self.search_paths:
            parts.append("\n  Searched in:")
            for path in self.search_paths:
                parts.append(f"\n    - {path}")

        return " ".join(parts) if not (self.cycle or self.search_paths) else "".join(parts)
