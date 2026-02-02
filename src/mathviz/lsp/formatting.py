"""
Code formatting for MathViz LSP.

This module provides document formatting functionality by wrapping
the existing MathViz formatter for LSP integration.
"""


from lsprotocol import types

from mathviz.formatter import FormatConfig, Formatter, format_source


class LSPFormatter:
    """
    Provides code formatting for the MathViz LSP server.

    Wraps the existing MathViz formatter to produce LSP-compatible
    text edits for document and range formatting.
    """

    def __init__(self, config: FormatConfig | None = None) -> None:
        """
        Initialize the formatter.

        Args:
            config: Optional formatting configuration
        """
        self.config = config or FormatConfig()
        self._formatter = Formatter(self.config)

    def format_document(self, source: str) -> list[types.TextEdit]:
        """
        Format an entire document.

        Args:
            source: The MathViz source code to format

        Returns:
            List of text edits to apply (typically one edit replacing the whole document)
        """
        try:
            formatted = format_source(source, self.config)

            # If no changes, return empty list
            if source == formatted:
                return []

            # Return a single edit replacing the entire document
            lines = source.splitlines()
            last_line = len(lines) - 1 if lines else 0
            last_char = len(lines[-1]) if lines else 0

            return [
                types.TextEdit(
                    range=types.Range(
                        start=types.Position(line=0, character=0),
                        end=types.Position(line=last_line, character=last_char),
                    ),
                    new_text=formatted.rstrip("\n"),  # LSP expects no trailing newline
                )
            ]

        except Exception:
            # If formatting fails (e.g., syntax error), return no edits
            # The error will be reported as a diagnostic
            return []

    def format_range(  # noqa: ARG002
        self,
        source: str,
        start_line: int,
        start_char: int,
        end_line: int,
        end_char: int,
    ) -> list[types.TextEdit]:
        """
        Format a range within a document.

        Note: Currently, this formats the entire document and returns
        edits. True range formatting would require more sophisticated
        AST manipulation.

        Args:
            source: The MathViz source code
            start_line: 0-indexed start line (unused, reserved for future)
            start_char: 0-indexed start character (unused, reserved for future)
            end_line: 0-indexed end line (unused, reserved for future)
            end_char: 0-indexed end character (unused, reserved for future)

        Returns:
            List of text edits to apply
        """
        # For now, format the entire document
        # A more sophisticated implementation would:
        # 1. Parse the document
        # 2. Find complete statements that overlap the range
        # 3. Format only those statements
        _ = (start_line, start_char, end_line, end_char)  # Reserved for future use
        return self.format_document(source)

    def format_on_type(
        self,
        source: str,
        line: int,
        character: int,
        trigger_char: str,
    ) -> list[types.TextEdit]:
        """
        Format after typing a trigger character.

        This provides incremental formatting as the user types.

        Args:
            source: The MathViz source code
            line: 0-indexed line where the character was typed
            character: 0-indexed character position
            trigger_char: The character that triggered formatting

        Returns:
            List of text edits to apply
        """
        lines = source.splitlines()

        if line >= len(lines):
            return []

        edits: list[types.TextEdit] = []

        if trigger_char == "}":
            # Auto-indent closing brace
            # Find matching opening brace and align
            edits.extend(self._format_closing_brace(lines, line, character))

        elif trigger_char == "\n":
            # Auto-indent new line based on previous line
            edits.extend(self._format_newline(lines, line))

        elif trigger_char == ";":
            # Potentially add newline after statement
            pass

        return edits

    def _format_closing_brace(
        self, lines: list[str], line: int, character: int  # noqa: ARG002
    ) -> list[types.TextEdit]:
        """
        Format a closing brace by aligning with its opening brace.

        Args:
            lines: All lines in the document
            line: The line with the closing brace
            character: Position of the closing brace

        Returns:
            Text edits to apply
        """
        # Find the matching opening brace
        brace_count = 1
        search_line = line - 1

        while search_line >= 0 and brace_count > 0:
            line_text = lines[search_line]
            for char in reversed(line_text):
                if char == "}":
                    brace_count += 1
                elif char == "{":
                    brace_count -= 1
                    if brace_count == 0:
                        break
            search_line -= 1

        if brace_count == 0:
            # Found matching brace
            opening_line = lines[search_line + 1]
            # Get the indentation of the opening line
            indent = len(opening_line) - len(opening_line.lstrip())

            current_line = lines[line]
            current_indent = len(current_line) - len(current_line.lstrip())

            if current_indent != indent:
                # Need to adjust indentation
                new_line = " " * indent + current_line.lstrip()
                return [
                    types.TextEdit(
                        range=types.Range(
                            start=types.Position(line=line, character=0),
                            end=types.Position(line=line, character=len(current_line)),
                        ),
                        new_text=new_line,
                    )
                ]

        return []

    def _format_newline(self, lines: list[str], line: int) -> list[types.TextEdit]:
        """
        Calculate appropriate indentation for a new line.

        Args:
            lines: All lines in the document
            line: The new line number

        Returns:
            Text edits to apply
        """
        if line == 0:
            return []

        prev_line = lines[line - 1]
        prev_indent = len(prev_line) - len(prev_line.lstrip())

        # Increase indent after opening brace or colon
        stripped = prev_line.rstrip()
        if stripped.endswith("{") or stripped.endswith(":"):
            new_indent = prev_indent + self.config.indent_size
        else:
            new_indent = prev_indent

        # If current line exists and needs adjustment
        if line < len(lines):
            current_line = lines[line]
            # Only adjust if the line is just whitespace or empty
            if not current_line.strip():
                indent_str = " " * new_indent if self.config.use_spaces else "\t" * (
                    new_indent // self.config.indent_size
                )
                return [
                    types.TextEdit(
                        range=types.Range(
                            start=types.Position(line=line, character=0),
                            end=types.Position(line=line, character=len(current_line)),
                        ),
                        new_text=indent_str,
                    )
                ]

        return []


def format_document(source: str, config: FormatConfig | None = None) -> list[types.TextEdit]:
    """
    Convenience function to format a document.

    Args:
        source: The MathViz source code
        config: Optional formatting configuration

    Returns:
        List of text edits
    """
    formatter = LSPFormatter(config)
    return formatter.format_document(source)


def format_range(
    source: str,
    start_line: int,
    start_char: int,
    end_line: int,
    end_char: int,
    config: FormatConfig | None = None,
) -> list[types.TextEdit]:
    """
    Convenience function to format a range.

    Args:
        source: The MathViz source code
        start_line: 0-indexed start line
        start_char: 0-indexed start character
        end_line: 0-indexed end line
        end_char: 0-indexed end character
        config: Optional formatting configuration

    Returns:
        List of text edits
    """
    formatter = LSPFormatter(config)
    return formatter.format_range(source, start_line, start_char, end_line, end_char)
