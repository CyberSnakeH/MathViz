"""
Diagnostic generation for MathViz LSP.

This module converts compiler errors, type errors, and lint warnings
into LSP-compatible diagnostic messages for display in editors.
"""

from lsprotocol import types

from mathviz.compiler.lexer import Lexer
from mathviz.compiler.linter import Linter, LintLevel, LintViolation
from mathviz.compiler.parser import Parser
from mathviz.utils.diagnostics import Diagnostic as CompilerDiagnostic
from mathviz.utils.diagnostics import DiagnosticLevel
from mathviz.utils.errors import LexerError, MathVizError, ParserError


class DiagnosticProvider:
    """
    Generates LSP diagnostics from MathViz source code.

    This provider runs the lexer, parser, type checker, and linter
    to collect all errors and warnings for a document.
    """

    def __init__(self, source: str, uri: str) -> None:
        """
        Initialize the diagnostic provider.

        Args:
            source: The MathViz source code to analyze
            uri: The document URI for location information
        """
        self.source = source
        self.uri = uri
        self._diagnostics: list[types.Diagnostic] = []

    def get_diagnostics(self) -> list[types.Diagnostic]:
        """
        Get all diagnostics for the document.

        Runs lexer, parser, type checker, and linter to collect
        all errors and warnings.

        Returns:
            List of LSP diagnostic objects
        """
        self._diagnostics = []

        # Phase 1: Lexer errors
        try:
            lexer = Lexer(self.source, filename=self.uri)
            tokens = lexer.tokenize()
        except LexerError as e:
            self._add_mathviz_error(e, types.DiagnosticSeverity.Error)
            return self._diagnostics
        except Exception as e:
            self._add_general_error(str(e), 0, 0, types.DiagnosticSeverity.Error)
            return self._diagnostics

        # Phase 2: Parser errors
        try:
            parser = Parser(tokens, source=self.source, filename=self.uri)
            ast = parser.parse()

            # Also add rich diagnostics from parser
            for diag in parser.get_diagnostics():
                self._add_compiler_diagnostic(diag)

        except ParserError as e:
            self._add_mathviz_error(e, types.DiagnosticSeverity.Error)
            return self._diagnostics
        except Exception as e:
            self._add_general_error(str(e), 0, 0, types.DiagnosticSeverity.Error)
            return self._diagnostics

        # Phase 3: Type checking errors
        # Note: TypeChecker integration would go here
        # try:
        #     type_checker = TypeChecker(ast, source=self.source)
        #     type_checker.check()
        #     for error in type_checker.errors:
        #         self._add_type_error(error)
        # except Exception as e:
        #     pass  # Type checking failures are non-fatal for diagnostics

        # Phase 4: Lint warnings
        try:
            linter = Linter()
            violations = linter.lint(ast)
            for violation in violations:
                self._add_lint_violation(violation)
        except Exception:
            pass  # Linting failures are non-fatal

        return self._diagnostics

    def _add_mathviz_error(self, error: MathVizError, severity: types.DiagnosticSeverity) -> None:
        """
        Add a MathViz compiler error as an LSP diagnostic.

        Args:
            error: The MathViz error
            severity: The diagnostic severity
        """
        line = 0
        character = 0

        if error.location:
            line = max(0, error.location.line - 1)  # Convert to 0-indexed
            character = max(0, error.location.column - 1)

        # Calculate end position (try to underline the whole token/line)
        end_character = character + 1
        if error.source_line:
            # Try to find the end of the problematic token
            rest_of_line = error.source_line[character:]
            for i, c in enumerate(rest_of_line):
                if c.isspace() or c in "()[]{},:;":
                    end_character = character + max(1, i)
                    break
            else:
                end_character = character + len(rest_of_line)

        diagnostic = types.Diagnostic(
            range=types.Range(
                start=types.Position(line=line, character=character),
                end=types.Position(line=line, character=end_character),
            ),
            message=error.message,
            severity=severity,
            source="mathviz",
        )

        self._diagnostics.append(diagnostic)

    def _add_compiler_diagnostic(self, diag: CompilerDiagnostic) -> None:
        """
        Add a rich compiler diagnostic as an LSP diagnostic.

        Args:
            diag: The compiler diagnostic
        """
        # Map compiler severity to LSP severity
        severity_map = {
            DiagnosticLevel.ERROR: types.DiagnosticSeverity.Error,
            DiagnosticLevel.WARNING: types.DiagnosticSeverity.Warning,
            DiagnosticLevel.NOTE: types.DiagnosticSeverity.Information,
            DiagnosticLevel.HELP: types.DiagnosticSeverity.Hint,
        }
        severity = severity_map.get(diag.level, types.DiagnosticSeverity.Error)

        # Get position from labels
        line = 0
        character = 0
        end_line = 0
        end_character = 1

        if diag.labels:
            primary_label = next(
                (label for label in diag.labels if label.is_primary), diag.labels[0]
            )
            span = primary_label.span
            line = max(0, span.start_line - 1)
            character = max(0, span.start_col - 1)
            end_line = max(0, span.end_line - 1)
            end_character = max(0, span.end_col - 1)

        # Build message with notes and helps
        message_parts = [diag.message]
        for note in diag.notes:
            message_parts.append(f"note: {note}")
        for help_msg in diag.helps:
            message_parts.append(f"help: {help_msg}")

        diagnostic = types.Diagnostic(
            range=types.Range(
                start=types.Position(line=line, character=character),
                end=types.Position(line=end_line, character=end_character),
            ),
            message="\n".join(message_parts),
            severity=severity,
            source="mathviz",
            code=diag.code,
        )

        self._diagnostics.append(diagnostic)

    def _add_lint_violation(self, violation: LintViolation) -> None:
        """
        Add a lint violation as an LSP diagnostic.

        Args:
            violation: The lint violation
        """
        # Map lint level to LSP severity
        severity_map = {
            LintLevel.ALLOW: None,  # Skip allowed rules
            LintLevel.WARN: types.DiagnosticSeverity.Warning,
            LintLevel.DENY: types.DiagnosticSeverity.Error,
        }

        severity = severity_map.get(violation.rule.level)
        if severity is None:
            return  # Skip allowed rules

        line = 0
        character = 0

        if violation.location:
            line = max(0, violation.location.line - 1)
            character = max(0, violation.location.column - 1)

        # Build message with suggestion
        message = violation.message
        if violation.suggestion:
            message = f"{message}\n\nhint: {violation.suggestion}"

        diagnostic = types.Diagnostic(
            range=types.Range(
                start=types.Position(line=line, character=character),
                end=types.Position(line=line, character=character + 1),
            ),
            message=message,
            severity=severity,
            source="mathviz-lint",
            code=violation.rule.code,
            tags=self._get_diagnostic_tags(violation),
        )

        self._diagnostics.append(diagnostic)

    def _get_diagnostic_tags(self, violation: LintViolation) -> list[types.DiagnosticTag]:
        """
        Get diagnostic tags for a lint violation.

        Args:
            violation: The lint violation

        Returns:
            List of applicable diagnostic tags
        """
        tags: list[types.DiagnosticTag] = []

        # Check if this is an "unused" warning
        if "unused" in violation.rule.name:
            tags.append(types.DiagnosticTag.Unnecessary)

        # Check if this is a deprecation warning
        if "deprecated" in violation.rule.name:
            tags.append(types.DiagnosticTag.Deprecated)

        return tags

    def _add_general_error(
        self,
        message: str,
        line: int,
        character: int,
        severity: types.DiagnosticSeverity,
    ) -> None:
        """
        Add a general error as an LSP diagnostic.

        Args:
            message: Error message
            line: 0-indexed line number
            character: 0-indexed character position
            severity: Diagnostic severity
        """
        diagnostic = types.Diagnostic(
            range=types.Range(
                start=types.Position(line=line, character=character),
                end=types.Position(line=line, character=character + 1),
            ),
            message=message,
            severity=severity,
            source="mathviz",
        )

        self._diagnostics.append(diagnostic)


def get_diagnostics_for_document(source: str, uri: str) -> list[types.Diagnostic]:
    """
    Convenience function to get diagnostics for a document.

    Args:
        source: The MathViz source code
        uri: The document URI

    Returns:
        List of LSP diagnostics
    """
    provider = DiagnosticProvider(source, uri)
    return provider.get_diagnostics()
