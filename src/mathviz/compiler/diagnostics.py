"""
Rust-like Smart Compiler Diagnostics for MathViz.

This module provides a comprehensive diagnostic system inspired by Rust's
famously helpful compiler error messages. It includes:

- Rich error messages with source context
- "Did you mean?" suggestions using Levenshtein distance
- Detailed help text and related diagnostics
- Beautiful terminal formatting with colors and underlines

Example output:
    error[E0001]: cannot find value `circel` in this scope
      --> example.mviz:5:12
       |
     5 |     let area = circel.radius * PI
       |                ^^^^^^ not found in this scope
       |
       = help: did you mean `circle`?
       = note: `circle` was defined at line 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mathviz.utils.errors import SourceLocation


# =============================================================================
# Diagnostic Severity Levels
# =============================================================================


class DiagnosticSeverity(Enum):
    """
    Severity level for diagnostic messages.

    Mirrors Rust's diagnostic levels for familiar developer experience.
    """

    ERROR = "error"  # Compilation cannot continue
    WARNING = "warning"  # Code compiles but may have issues
    INFO = "info"  # Informational message
    HINT = "hint"  # Suggestion for improvement

    def color_code(self) -> str:
        """Get ANSI color code for terminal output."""
        colors = {
            DiagnosticSeverity.ERROR: "\033[91m",  # Red
            DiagnosticSeverity.WARNING: "\033[93m",  # Yellow
            DiagnosticSeverity.INFO: "\033[96m",  # Cyan
            DiagnosticSeverity.HINT: "\033[92m",  # Green
        }
        return colors.get(self, "")

    @property
    def label(self) -> str:
        """Get the label for this severity."""
        return self.value


# =============================================================================
# Core Diagnostic Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """
    A span of source code for highlighting.

    Represents a range in the source that should be highlighted
    or annotated in error messages.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    filename: str = "<input>"

    @classmethod
    def from_location(
        cls,
        loc: SourceLocation,
        length: int = 1,
    ) -> "SourceSpan":
        """Create a span from a SourceLocation."""
        return cls(
            start_line=loc.line,
            start_col=loc.column,
            end_line=loc.line,
            end_col=loc.column + length,
            filename=loc.filename or "<input>",
        )

    @classmethod
    def single_char(cls, line: int, col: int, filename: str = "<input>") -> "SourceSpan":
        """Create a single-character span."""
        return cls(line, col, line, col + 1, filename)

    @property
    def is_multiline(self) -> bool:
        """Check if this span covers multiple lines."""
        return self.start_line != self.end_line

    @property
    def length(self) -> int:
        """Get the length on a single line."""
        if self.is_multiline:
            return 1
        return max(1, self.end_col - self.start_col)

    def __str__(self) -> str:
        return f"{self.filename}:{self.start_line}:{self.start_col}"


@dataclass
class Diagnostic:
    """
    A rich diagnostic message with source context and suggestions.

    This is the primary class for representing compiler errors, warnings,
    and other diagnostic messages in a Rust-like format.

    Attributes:
        severity: The severity level (ERROR, WARNING, INFO, HINT)
        code: Error code like "E0001", "W0001", etc.
        message: The primary diagnostic message
        location: Source location where the diagnostic occurred
        suggestion: Optional "did you mean X?" suggestion
        help_text: Optional longer help explanation
        related: List of related diagnostics (e.g., "defined here")
        labels: Additional source spans to highlight
    """

    severity: DiagnosticSeverity
    code: str
    message: str
    location: Optional[SourceLocation] = None
    span: Optional[SourceSpan] = None
    suggestion: Optional[str] = None
    help_text: Optional[str] = None
    related: list["Diagnostic"] = field(default_factory=list)
    labels: list[tuple[SourceSpan, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize related list if None."""
        if self.related is None:
            object.__setattr__(self, "related", [])
        if self.labels is None:
            object.__setattr__(self, "labels", [])
        if self.notes is None:
            object.__setattr__(self, "notes", [])

    def add_related(self, diagnostic: "Diagnostic") -> "Diagnostic":
        """Add a related diagnostic and return self for chaining."""
        self.related.append(diagnostic)
        return self

    def add_label(self, span: SourceSpan, message: str = "") -> "Diagnostic":
        """Add a source label and return self for chaining."""
        self.labels.append((span, message))
        return self

    def add_note(self, note: str) -> "Diagnostic":
        """Add a note and return self for chaining."""
        self.notes.append(note)
        return self

    def with_suggestion(self, suggestion: str) -> "Diagnostic":
        """Set suggestion and return self for chaining."""
        self.suggestion = suggestion
        return self

    def with_help(self, help_text: str) -> "Diagnostic":
        """Set help text and return self for chaining."""
        self.help_text = help_text
        return self


# =============================================================================
# Levenshtein Distance for "Did you mean?" Suggestions
# =============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    The edit distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) needed to transform s1 into s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The edit distance as an integer

    Example:
        >>> levenshtein_distance("circle", "circel")
        2
        >>> levenshtein_distance("cat", "dog")
        3
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use Wagner-Fischer algorithm with O(n) space
    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_similar_names(
    name: str,
    candidates: list[str],
    max_distance: int = 2,
    max_suggestions: int = 3,
) -> list[str]:
    """
    Find similar names from a list of candidates for suggestions.

    Uses Levenshtein distance to identify names that are close to the
    given name, enabling "did you mean?" suggestions.

    Args:
        name: The name to find suggestions for
        candidates: List of valid names to compare against
        max_distance: Maximum edit distance to consider (default 2)
        max_suggestions: Maximum number of suggestions to return (default 3)

    Returns:
        List of similar names, sorted by similarity (closest first)

    Example:
        >>> find_similar_names("circel", ["circle", "square", "triangle"])
        ['circle']
    """
    if not candidates:
        return []

    # Score candidates by edit distance
    scored: list[tuple[int, str]] = []
    for candidate in candidates:
        # Skip if lengths are too different
        if abs(len(candidate) - len(name)) > max_distance:
            continue

        distance = levenshtein_distance(name.lower(), candidate.lower())
        if distance <= max_distance:
            scored.append((distance, candidate))

    # Sort by distance (ascending), then alphabetically
    scored.sort(key=lambda x: (x[0], x[1]))

    return [candidate for _, candidate in scored[:max_suggestions]]


def find_best_match(name: str, candidates: list[str], max_distance: int = 2) -> Optional[str]:
    """
    Find the single best matching name from candidates.

    Args:
        name: The name to match
        candidates: List of valid names
        max_distance: Maximum edit distance

    Returns:
        The best match or None if no match is close enough
    """
    similar = find_similar_names(name, candidates, max_distance, 1)
    return similar[0] if similar else None


# =============================================================================
# Diagnostic Emitter
# =============================================================================


class DiagnosticEmitter:
    """
    Collects and formats diagnostics like Rust's compiler.

    The emitter provides factory methods for common diagnostic types
    and handles formatting them with source context.

    Example:
        emitter = DiagnosticEmitter(source_code, "example.mviz")

        # Emit an undefined variable error
        emitter.emit_undefined_variable("circel", location, ["circle", "center"])

        # Format and print all diagnostics
        for diag in emitter.diagnostics:
            print(emitter.format_diagnostic(diag))
    """

    def __init__(
        self,
        source: str = "",
        filename: str = "<input>",
    ) -> None:
        """
        Initialize the diagnostic emitter.

        Args:
            source: The source code being compiled
            filename: The filename for error reporting
        """
        self.source = source
        self.source_lines = source.splitlines() if source else []
        self.filename = filename
        self.diagnostics: list[Diagnostic] = []

    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        """Add a diagnostic and return it."""
        self.diagnostics.append(diagnostic)
        return diagnostic

    def emit_undefined_variable(
        self,
        name: str,
        loc: SourceLocation,
        candidates: list[str],
        defined_at: Optional[SourceLocation] = None,
    ) -> Diagnostic:
        """
        Emit an undefined variable error with 'did you mean?' suggestion.

        Args:
            name: The undefined variable name
            loc: Location where it was used
            candidates: List of valid variable names for suggestions
            defined_at: Optional location where a suggested variable was defined

        Returns:
            The created Diagnostic
        """
        similar = find_similar_names(name, candidates)
        suggestion = f"did you mean `{similar[0]}`?" if similar else None

        diag = Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            code="E0001",
            message=f"cannot find value `{name}` in this scope",
            location=loc,
            span=SourceSpan.from_location(loc, len(name)),
            suggestion=suggestion,
        )

        # Add related diagnostic if we know where similar was defined
        if similar and defined_at:
            diag.add_note(f"`{similar[0]}` was defined at line {defined_at.line}")

        return self.add(diag)

    def emit_type_mismatch(
        self,
        expected: str,
        found: str,
        loc: SourceLocation,
        context: Optional[str] = None,
    ) -> Diagnostic:
        """
        Emit a type mismatch error with conversion help.

        Args:
            expected: The expected type name
            found: The actual type found
            loc: Source location
            context: Optional context (e.g., "in assignment to 'x'")

        Returns:
            The created Diagnostic
        """
        message = f"mismatched types: expected `{expected}`, found `{found}`"
        if context:
            message = f"{message} {context}"

        # Generate help text for common conversions
        help_text = None
        if expected == "Float" and found == "Int":
            help_text = "use `float(value)` to convert Int to Float"
        elif expected == "Int" and found == "Float":
            help_text = "use `int(value)` to convert Float to Int (truncates decimal)"
        elif expected == "String" and found in ("Int", "Float", "Bool"):
            help_text = f"use `str(value)` to convert {found} to String"
        elif expected == "Bool" and found in ("Int", "Float"):
            help_text = "non-zero numbers are truthy; use comparison for explicit boolean"
        else:
            help_text = f"consider converting `{found}` to `{expected}`"

        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="E0002",
                message=message,
                location=loc,
                span=SourceSpan.from_location(loc),
                help_text=help_text,
            )
        )

    def emit_undefined_function(
        self,
        name: str,
        loc: SourceLocation,
        candidates: list[str],
    ) -> Diagnostic:
        """Emit an undefined function error with suggestions."""
        similar = find_similar_names(name, candidates)
        suggestion = f"did you mean `{similar[0]}`?" if similar else None

        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="E0003",
                message=f"cannot find function `{name}` in this scope",
                location=loc,
                span=SourceSpan.from_location(loc, len(name)),
                suggestion=suggestion,
            )
        )

    def emit_wrong_arg_count(
        self,
        func_name: str,
        expected: int,
        found: int,
        loc: SourceLocation,
    ) -> Diagnostic:
        """Emit a wrong argument count error."""
        if found < expected:
            help_text = f"add {expected - found} more argument(s)"
        else:
            help_text = f"remove {found - expected} argument(s)"

        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="E0004",
                message=f"function `{func_name}` expects {expected} argument(s), found {found}",
                location=loc,
                span=SourceSpan.from_location(loc),
                help_text=help_text,
            )
        )

    def emit_unused_variable(
        self,
        name: str,
        loc: SourceLocation,
    ) -> Diagnostic:
        """Emit an unused variable warning."""
        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0001",
                message=f"unused variable: `{name}`",
                location=loc,
                span=SourceSpan.from_location(loc, len(name)),
                suggestion=f"prefix with `_` to silence this warning: `_{name}`",
            )
        )

    def emit_unused_function(
        self,
        name: str,
        loc: SourceLocation,
    ) -> Diagnostic:
        """Emit an unused function warning."""
        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0002",
                message=f"function `{name}` is never called",
                location=loc,
                span=SourceSpan.from_location(loc, len(name)),
                suggestion="remove the function or add a call to it",
            )
        )

    def emit_unreachable_code(
        self,
        loc: SourceLocation,
        reason: str = "code after this statement is unreachable",
    ) -> Diagnostic:
        """Emit an unreachable code warning."""
        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0003",
                message=reason,
                location=loc,
                span=SourceSpan.from_location(loc),
                suggestion="remove the unreachable code",
            )
        )

    def emit_non_exhaustive_match(
        self,
        loc: SourceLocation,
        missing_patterns: list[str],
    ) -> Diagnostic:
        """Emit a non-exhaustive match warning."""
        if len(missing_patterns) == 1:
            pattern_str = f"`{missing_patterns[0]}`"
        else:
            pattern_str = ", ".join(f"`{p}`" for p in missing_patterns[:-1])
            pattern_str += f" and `{missing_patterns[-1]}`"

        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0004",
                message=f"non-exhaustive match: patterns {pattern_str} not covered",
                location=loc,
                span=SourceSpan.from_location(loc),
                help_text="add a wildcard pattern `_` to handle remaining cases",
            )
        )

    def emit_unreachable_pattern(
        self,
        loc: SourceLocation,
        previous_loc: SourceLocation,
    ) -> Diagnostic:
        """Emit an unreachable pattern warning."""
        diag = Diagnostic(
            severity=DiagnosticSeverity.WARNING,
            code="W0005",
            message="unreachable pattern (previous pattern catches all)",
            location=loc,
            span=SourceSpan.from_location(loc),
            suggestion="remove this pattern or reorder match arms",
        )
        diag.add_note(f"previous catch-all pattern at line {previous_loc.line}")
        return self.add(diag)

    def emit_redundant_pattern(
        self,
        loc: SourceLocation,
    ) -> Diagnostic:
        """Emit a redundant pattern warning."""
        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0006",
                message="redundant pattern (already covered by previous patterns)",
                location=loc,
                span=SourceSpan.from_location(loc),
                suggestion="remove this redundant pattern",
            )
        )

    def emit_unused_import(
        self,
        name: str,
        loc: SourceLocation,
    ) -> Diagnostic:
        """Emit an unused import warning."""
        return self.add(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="W0007",
                message=f"unused import: `{name}`",
                location=loc,
                span=SourceSpan.from_location(loc, len(name)),
                suggestion="remove the unused import",
            )
        )

    def emit_shadowing(
        self,
        name: str,
        loc: SourceLocation,
        original_loc: SourceLocation,
    ) -> Diagnostic:
        """Emit a variable shadowing warning."""
        diag = Diagnostic(
            severity=DiagnosticSeverity.WARNING,
            code="W0008",
            message=f"variable `{name}` shadows a variable from outer scope",
            location=loc,
            span=SourceSpan.from_location(loc, len(name)),
            suggestion="use a different variable name to avoid confusion",
        )
        diag.add_note(f"previously defined at line {original_loc.line}")
        return self.add(diag)

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_diagnostic(
        self,
        diag: Diagnostic,
        source_lines: Optional[list[str]] = None,
        use_color: bool = True,
    ) -> str:
        """
        Format a diagnostic like Rust's compiler output.

        Example output:
            error[E0001]: cannot find value `circel` in this scope
              --> example.mviz:5:12
               |
             5 |     let area = circel.radius * PI
               |                ^^^^^^ not found in this scope
               |
               = help: did you mean `circle`?
               = note: `circle` was defined at line 2

        Args:
            diag: The diagnostic to format
            source_lines: Source lines (uses self.source_lines if not provided)
            use_color: Whether to use ANSI color codes

        Returns:
            Formatted multi-line string
        """
        lines: list[str] = []
        src_lines = source_lines or self.source_lines

        # ANSI codes
        reset = "\033[0m" if use_color else ""
        bold = "\033[1m" if use_color else ""
        blue = "\033[94m" if use_color else ""
        severity_color = diag.severity.color_code() if use_color else ""

        # Header: error[E0001]: cannot find value `circel` in this scope
        header = (
            f"{severity_color}{bold}{diag.severity.label}[{diag.code}]{reset}: "
            f"{bold}{diag.message}{reset}"
        )
        lines.append(header)

        # Location: --> example.mviz:5:12
        if diag.location:
            loc = diag.location
            filename = loc.filename or self.filename
            lines.append(f"  {blue}-->{reset} {filename}:{loc.line}:{loc.column}")

        # Source context with underline
        if diag.location and src_lines:
            line_num = diag.location.line
            if 1 <= line_num <= len(src_lines):
                lines.append(f"   {blue}|{reset}")

                # Show the source line
                source_line = src_lines[line_num - 1]
                line_num_str = f"{line_num:3}"
                lines.append(f"{blue}{line_num_str} |{reset} {source_line}")

                # Underline the problematic span
                if diag.span:
                    col = diag.span.start_col
                    length = diag.span.length
                else:
                    col = diag.location.column
                    length = 1

                padding = " " * (col - 1)
                underline = "^" * length
                lines.append(f"   {blue}|{reset} {padding}{severity_color}{underline}{reset}")

                lines.append(f"   {blue}|{reset}")

        # Additional labels
        for span, label_msg in diag.labels:
            if src_lines and 1 <= span.start_line <= len(src_lines):
                source_line = src_lines[span.start_line - 1]
                line_num_str = f"{span.start_line:3}"
                lines.append(f"{blue}{line_num_str} |{reset} {source_line}")

                padding = " " * (span.start_col - 1)
                underline = "-" * span.length
                label_line = f"   {blue}|{reset} {padding}{blue}{underline}{reset}"
                if label_msg:
                    label_line += f" {blue}{label_msg}{reset}"
                lines.append(label_line)

        # Help text
        if diag.help_text:
            green = "\033[92m" if use_color else ""
            lines.append(f"   {blue}={reset} {green}help:{reset} {diag.help_text}")

        # Suggestion
        if diag.suggestion:
            green = "\033[92m" if use_color else ""
            lines.append(f"   {blue}={reset} {green}help:{reset} {diag.suggestion}")

        # Notes
        for note in diag.notes:
            lines.append(f"   {blue}={reset} {bold}note:{reset} {note}")

        # Related diagnostics
        for related in diag.related:
            lines.append("")
            lines.append(self.format_diagnostic(related, src_lines, use_color))

        return "\n".join(lines)

    def format_all(self, use_color: bool = True) -> str:
        """Format all diagnostics as a single string."""
        formatted: list[str] = []
        for diag in self.diagnostics:
            formatted.append(self.format_diagnostic(diag, use_color=use_color))
        return "\n\n".join(formatted)

    def has_errors(self) -> bool:
        """Check if any error diagnostics have been emitted."""
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def error_count(self) -> int:
        """Count error diagnostics."""
        return sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR)

    def warning_count(self) -> int:
        """Count warning diagnostics."""
        return sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING)

    def clear(self) -> None:
        """Clear all diagnostics."""
        self.diagnostics.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def format_error(
    message: str,
    code: str,
    loc: SourceLocation,
    source: str,
    suggestion: Optional[str] = None,
) -> str:
    """
    Format a single error message with source context.

    Convenience function for quick error formatting without
    creating a full DiagnosticEmitter.
    """
    emitter = DiagnosticEmitter(source)
    diag = Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=code,
        message=message,
        location=loc,
        span=SourceSpan.from_location(loc),
        suggestion=suggestion,
    )
    return emitter.format_diagnostic(diag)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core types
    "DiagnosticSeverity",
    "SourceSpan",
    "Diagnostic",
    "DiagnosticEmitter",
    # Similarity functions
    "levenshtein_distance",
    "find_similar_names",
    "find_best_match",
    # Convenience functions
    "format_error",
]
