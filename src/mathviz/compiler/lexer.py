"""
MathViz Lexer (Tokenizer).

Transforms MathViz source code into a stream of tokens, supporting both
standard programming constructs and mathematical Unicode symbols.
"""

from typing import Iterator, Optional

from mathviz.compiler.tokens import (
    Token,
    TokenType,
    KEYWORDS,
    MATH_SYMBOLS,
    SINGLE_CHAR_TOKENS,
    DOUBLE_CHAR_TOKENS,
)

# Pending doc comment to attach to the next statement
_pending_doc_comment: str | None = None
from mathviz.utils.errors import LexerError, SourceLocation


class Lexer:
    """
    Tokenizer for MathViz source code.

    The lexer supports:
    - Standard identifiers and keywords
    - Integer and floating-point literals
    - String literals (single and double quoted)
    - Mathematical Unicode operators (∈, ∪, ⊆, etc.)
    - Comments (# single line, /* multi-line */)
    - Block-based syntax with curly braces

    Usage:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        # or iterate: for token in lexer: ...
    """

    def __init__(self, source: str, filename: Optional[str] = None) -> None:
        """
        Initialize the lexer with source code.

        Args:
            source: The MathViz source code to tokenize
            filename: Optional filename for error reporting
        """
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

        # Track the start of the current line for error reporting
        self._line_start = 0

    @property
    def _current_char(self) -> Optional[str]:
        """Return the current character or None if at end."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    @property
    def _peek_char(self) -> Optional[str]:
        """Return the next character without consuming it."""
        peek_pos = self.pos + 1
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]

    def _peek_ahead(self, n: int) -> Optional[str]:
        """Return the character n positions ahead."""
        peek_pos = self.pos + n
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]

    def _location(self) -> SourceLocation:
        """Create a SourceLocation for the current position."""
        return SourceLocation(
            line=self.line,
            column=self.column,
            offset=self.pos,
            filename=self.filename,
        )

    def _current_line_text(self) -> str:
        """Extract the current line of source for error messages."""
        end = self.source.find("\n", self._line_start)
        if end == -1:
            end = len(self.source)
        return self.source[self._line_start:end]

    def _advance(self) -> str:
        """Consume and return the current character."""
        char = self.source[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
            self._line_start = self.pos
        else:
            self.column += 1

        return char

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters except newlines."""
        while self._current_char is not None and self._current_char in " \t\r":
            self._advance()

    def _skip_comment(self) -> None:
        """Skip single-line comments starting with #."""
        if self._current_char == "#":
            while self._current_char is not None and self._current_char != "\n":
                self._advance()

    def _skip_double_slash_comment(self) -> bool:
        """Skip single-line comments starting with //.

        Returns True if a comment was skipped.
        """
        if self._current_char == "/" and self._peek_char == "/":
            # Make sure it's not a doc comment (///)
            if self._peek_ahead(2) == "/":
                return False  # Let doc comment handler deal with it
            # Skip the //
            self._advance()
            self._advance()
            # Skip until end of line
            while self._current_char is not None and self._current_char != "\n":
                self._advance()
            return True
        return False

    def _read_doc_comment(self) -> Token | None:
        """
        Read a documentation comment.

        Handles:
            /// Single line doc comment
            /** Multi-line
                doc comment */

        Returns:
            A DOC_COMMENT token, or None if not a doc comment.
        """
        if self._current_char != "/":
            return None

        start_loc = self._location()

        # Check for /// single line doc comment
        if self._peek_char == "/" and self._peek_ahead(2) == "/":
            self._advance()  # first /
            self._advance()  # second /
            self._advance()  # third /

            # Collect comment content until end of line
            content_chars: list[str] = []

            # Skip leading space after ///
            if self._current_char == " ":
                self._advance()

            while self._current_char is not None and self._current_char != "\n":
                content_chars.append(self._current_char)
                self._advance()

            # Check for continuation on next lines
            content = "".join(content_chars).rstrip()

            # Look ahead for more /// lines
            while True:
                # Save position to backtrack if not a continuation
                saved_pos = self.pos
                saved_line = self.line
                saved_col = self.column

                # Skip the newline
                if self._current_char == "\n":
                    self._advance()

                # Skip whitespace on new line
                while self._current_char is not None and self._current_char in " \t":
                    self._advance()

                # Check for another ///
                if (self._current_char == "/" and
                    self._peek_char == "/" and
                    self._peek_ahead(2) == "/"):
                    self._advance()  # first /
                    self._advance()  # second /
                    self._advance()  # third /

                    # Skip leading space
                    if self._current_char == " ":
                        self._advance()

                    # Collect this line
                    line_chars: list[str] = []
                    while self._current_char is not None and self._current_char != "\n":
                        line_chars.append(self._current_char)
                        self._advance()

                    content += "\n" + "".join(line_chars).rstrip()
                else:
                    # Not a continuation, backtrack
                    self.pos = saved_pos
                    self.line = saved_line
                    self.column = saved_col
                    break

            return Token(TokenType.DOC_COMMENT, content, start_loc)

        # Check for /** multi-line doc comment */
        if self._peek_char == "*" and self._peek_ahead(2) == "*":
            # This is /**, not just /*
            self._advance()  # /
            self._advance()  # first *
            self._advance()  # second *

            content_chars: list[str] = []

            # Skip leading whitespace/newline
            while self._current_char is not None and self._current_char in " \t\n":
                if self._current_char == "\n":
                    self._advance()
                    # Skip leading * on continuation lines
                    while self._current_char is not None and self._current_char in " \t":
                        self._advance()
                    if self._current_char == "*" and self._peek_char != "/":
                        self._advance()
                        if self._current_char == " ":
                            self._advance()
                    break
                self._advance()

            while True:
                if self._current_char is None:
                    raise LexerError(
                        "Unterminated doc comment",
                        start_loc,
                        self._current_line_text(),
                    )

                # Check for closing */
                if self._current_char == "*" and self._peek_char == "/":
                    self._advance()  # *
                    self._advance()  # /
                    break

                # Handle newlines - clean up leading * on continuation lines
                if self._current_char == "\n":
                    content_chars.append("\n")
                    self._advance()

                    # Skip leading whitespace
                    while self._current_char is not None and self._current_char in " \t":
                        self._advance()

                    # Skip leading * (common doc comment style)
                    if self._current_char == "*" and self._peek_char != "/":
                        self._advance()
                        if self._current_char == " ":
                            self._advance()
                    continue

                content_chars.append(self._current_char)
                self._advance()

            content = "".join(content_chars).strip()
            return Token(TokenType.DOC_COMMENT, content, start_loc)

        return None

    def _skip_multiline_comment(self) -> bool:
        """
        Skip multi-line comments /* ... */.

        Returns:
            True if a multi-line comment was skipped, False otherwise.
        """
        if self._current_char == "/" and self._peek_char == "*":
            start_loc = self._location()
            self._advance()  # /
            self._advance()  # *

            while True:
                if self._current_char is None:
                    raise LexerError(
                        "Unterminated multi-line comment",
                        start_loc,
                        self._current_line_text(),
                    )
                if self._current_char == "*" and self._peek_char == "/":
                    self._advance()  # *
                    self._advance()  # /
                    return True
                self._advance()

        return False

    def _read_string(self, quote_char: str) -> Token:
        """
        Read a string literal.

        Args:
            quote_char: The opening quote character (' or ")

        Returns:
            A STRING token with the string value.
        """
        start_loc = self._location()
        self._advance()  # consume opening quote

        value_chars: list[str] = []
        escape_sequences = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "\\": "\\",
            "'": "'",
            '"': '"',
            "0": "\0",
        }

        while True:
            if self._current_char is None:
                raise LexerError(
                    "Unterminated string literal",
                    start_loc,
                    self._current_line_text(),
                )

            if self._current_char == "\n":
                raise LexerError(
                    "Newline in string literal (use \\n for newlines)",
                    self._location(),
                    self._current_line_text(),
                )

            if self._current_char == quote_char:
                self._advance()  # consume closing quote
                break

            if self._current_char == "\\":
                self._advance()
                if self._current_char is None:
                    raise LexerError(
                        "Unterminated escape sequence",
                        self._location(),
                        self._current_line_text(),
                    )
                escaped = escape_sequences.get(self._current_char)
                if escaped is None:
                    raise LexerError(
                        f"Invalid escape sequence: \\{self._current_char}",
                        self._location(),
                        self._current_line_text(),
                    )
                value_chars.append(escaped)
                self._advance()
            else:
                value_chars.append(self._current_char)
                self._advance()

        return Token(TokenType.STRING, "".join(value_chars), start_loc)

    def _read_number(self) -> Token:
        """
        Read a numeric literal (integer or float).

        Supports:
        - Decimal integers: 123
        - Floats: 123.456
        - Scientific notation: 1.23e10, 1.23E-10
        - Underscores for readability: 1_000_000

        Returns:
            An INTEGER or FLOAT token.
        """
        start_loc = self._location()
        num_chars: list[str] = []
        is_float = False

        # Read integer part
        while self._current_char is not None and (
            self._current_char.isdigit() or self._current_char == "_"
        ):
            if self._current_char != "_":
                num_chars.append(self._current_char)
            self._advance()

        # Check for decimal point
        if self._current_char == "." and (
            self._peek_char is not None and self._peek_char.isdigit()
        ):
            is_float = True
            num_chars.append(self._current_char)
            self._advance()

            # Read fractional part
            while self._current_char is not None and (
                self._current_char.isdigit() or self._current_char == "_"
            ):
                if self._current_char != "_":
                    num_chars.append(self._current_char)
                self._advance()

        # Check for exponent
        if self._current_char is not None and self._current_char in "eE":
            is_float = True
            num_chars.append(self._current_char)
            self._advance()

            # Optional sign
            if self._current_char is not None and self._current_char in "+-":
                num_chars.append(self._current_char)
                self._advance()

            # Exponent digits (required)
            if self._current_char is None or not self._current_char.isdigit():
                raise LexerError(
                    "Invalid number: expected exponent digits",
                    self._location(),
                    self._current_line_text(),
                )

            while self._current_char is not None and self._current_char.isdigit():
                num_chars.append(self._current_char)
                self._advance()

        value_str = "".join(num_chars)

        if is_float:
            return Token(TokenType.FLOAT, float(value_str), start_loc)
        return Token(TokenType.INTEGER, int(value_str), start_loc)

    def _read_identifier_or_keyword(self) -> Token:
        """
        Read an identifier or keyword.

        Identifiers start with a letter or underscore and contain
        letters, digits, and underscores.

        Returns:
            An IDENTIFIER token or the appropriate keyword token.
        """
        start_loc = self._location()
        id_chars: list[str] = []

        while self._current_char is not None and (
            self._current_char.isalnum() or self._current_char == "_"
        ):
            id_chars.append(self._current_char)
            self._advance()

        identifier = "".join(id_chars)

        # Check if it's a keyword
        if identifier in KEYWORDS:
            token_type = KEYWORDS[identifier]
            # Handle boolean literals specially
            if token_type == TokenType.TRUE:
                return Token(token_type, True, start_loc)
            if token_type == TokenType.FALSE:
                return Token(token_type, False, start_loc)
            if token_type == TokenType.NONE:
                return Token(token_type, None, start_loc)
            return Token(token_type, identifier, start_loc)

        return Token(TokenType.IDENTIFIER, identifier, start_loc)

    def _read_math_symbol(self) -> Optional[Token]:
        """
        Try to read a mathematical Unicode symbol.

        Returns:
            A token for the math symbol, or None if not a recognized symbol.
        """
        char = self._current_char
        if char is not None and char in MATH_SYMBOLS:
            start_loc = self._location()
            token_type = MATH_SYMBOLS[char]
            self._advance()

            # Handle special math symbol values
            if token_type == TokenType.PI:
                return Token(token_type, 3.141592653589793, start_loc)
            if token_type == TokenType.INFINITY:
                return Token(token_type, float("inf"), start_loc)
            if token_type == TokenType.EMPTY_SET:
                return Token(token_type, set(), start_loc)

            return Token(token_type, char, start_loc)

        return None

    def _read_fstring(self) -> Token:
        """
        Read an f-string literal.

        F-strings support expression interpolation with {expr} syntax.
        Format specifiers can be added with {expr:format}.
        Nested braces {{ and }} represent literal braces.

        Returns:
            An FSTRING_START token. The lexer will tokenize parts and expressions
            as separate tokens in subsequent calls.
        """
        start_loc = self._location()
        self._advance()  # consume 'f'
        quote_char = self._current_char
        self._advance()  # consume opening quote

        # We'll emit a special STRING token with fstring metadata
        # The token value will be a list of (type, value) tuples
        parts: list[tuple[str, str]] = []
        current_literal: list[str] = []

        while True:
            if self._current_char is None:
                raise LexerError(
                    "Unterminated f-string literal",
                    start_loc,
                    self._current_line_text(),
                )

            if self._current_char == "\n":
                raise LexerError(
                    "Newline in f-string literal (use \\n for newlines)",
                    self._location(),
                    self._current_line_text(),
                )

            # End of f-string
            if self._current_char == quote_char:
                # Save any accumulated literal
                if current_literal:
                    parts.append(("literal", "".join(current_literal)))
                self._advance()  # consume closing quote
                break

            # Escaped characters
            if self._current_char == "\\":
                self._advance()
                if self._current_char is None:
                    raise LexerError(
                        "Unterminated escape sequence in f-string",
                        self._location(),
                        self._current_line_text(),
                    )
                escape_sequences = {
                    "n": "\n",
                    "t": "\t",
                    "r": "\r",
                    "\\": "\\",
                    "'": "'",
                    '"': '"',
                    "0": "\0",
                    "{": "{",  # Escaped brace
                    "}": "}",  # Escaped brace
                }
                escaped = escape_sequences.get(self._current_char)
                if escaped is None:
                    raise LexerError(
                        f"Invalid escape sequence in f-string: \\{self._current_char}",
                        self._location(),
                        self._current_line_text(),
                    )
                current_literal.append(escaped)
                self._advance()
                continue

            # Double braces for literal braces
            if self._current_char == "{" and self._peek_char == "{":
                current_literal.append("{")
                self._advance()  # first {
                self._advance()  # second {
                continue

            if self._current_char == "}" and self._peek_char == "}":
                current_literal.append("}")
                self._advance()  # first }
                self._advance()  # second }
                continue

            # Expression interpolation
            if self._current_char == "{":
                # Save accumulated literal before expression
                if current_literal:
                    parts.append(("literal", "".join(current_literal)))
                    current_literal = []

                self._advance()  # consume '{'

                # Read expression and optional format spec
                expr_chars: list[str] = []
                format_spec: list[str] = []
                brace_depth = 1
                in_format_spec = False

                while brace_depth > 0:
                    if self._current_char is None:
                        raise LexerError(
                            "Unterminated expression in f-string",
                            self._location(),
                            self._current_line_text(),
                        )

                    if self._current_char == "{":
                        brace_depth += 1
                        if in_format_spec:
                            format_spec.append(self._current_char)
                        else:
                            expr_chars.append(self._current_char)
                    elif self._current_char == "}":
                        brace_depth -= 1
                        if brace_depth > 0:
                            if in_format_spec:
                                format_spec.append(self._current_char)
                            else:
                                expr_chars.append(self._current_char)
                    elif self._current_char == ":" and brace_depth == 1 and not in_format_spec:
                        # Start of format specifier
                        in_format_spec = True
                    elif in_format_spec:
                        format_spec.append(self._current_char)
                    else:
                        expr_chars.append(self._current_char)

                    if brace_depth > 0:  # Don't advance past closing brace
                        self._advance()

                self._advance()  # consume closing '}'

                expr_str = "".join(expr_chars).strip()
                format_str = "".join(format_spec) if format_spec else None
                if format_str:
                    parts.append(("expr", expr_str, format_str))
                else:
                    parts.append(("expr", expr_str))
                continue

            # Regular character
            current_literal.append(self._current_char)
            self._advance()

        # Return a STRING token with fstring metadata
        # The value is a tuple: ("fstring", parts)
        return Token(TokenType.STRING, ("fstring", tuple(parts)), start_loc)

    def _read_operator(self) -> Optional[Token]:
        """
        Read an operator token (single or double character).

        Returns:
            An operator token, or None if the current character is not an operator.
        """
        if self._current_char is None:
            return None

        start_loc = self._location()

        # Try three-character operators first (...)
        if (self._current_char == "." and
            self._peek_char == "." and
            self._peek_ahead(2) == "."):
            self._advance()
            self._advance()
            self._advance()
            return Token(TokenType.ELLIPSIS, "...", start_loc)

        # Try two-character operators
        if self._peek_char is not None:
            two_char = self._current_char + self._peek_char
            if two_char in DOUBLE_CHAR_TOKENS:
                token_type = DOUBLE_CHAR_TOKENS[two_char]
                self._advance()
                self._advance()
                return Token(token_type, two_char, start_loc)

        # Try single-character operators
        if self._current_char in SINGLE_CHAR_TOKENS:
            char = self._current_char
            token_type = SINGLE_CHAR_TOKENS[char]
            self._advance()
            return Token(token_type, char, start_loc)

        return None

    def _next_token(self) -> Optional[Token]:
        """
        Extract the next token from the source.

        Returns:
            The next token, or None if at end of source.
        """
        # Skip whitespace and comments (but capture doc comments)
        while True:
            self._skip_whitespace()

            # Check for doc comments (/// or /**) BEFORE regular comments
            if self._current_char == "/":
                # Check for /// doc comment
                if self._peek_char == "/" and self._peek_ahead(2) == "/":
                    doc_token = self._read_doc_comment()
                    if doc_token:
                        return doc_token
                # Check for /** doc comment (not just /*)
                elif self._peek_char == "*" and self._peek_ahead(2) == "*":
                    doc_token = self._read_doc_comment()
                    if doc_token:
                        return doc_token

            # Check for // single-line comments (before treating / as operator)
            if self._skip_double_slash_comment():
                continue

            if self._current_char == "#":
                self._skip_comment()
                continue

            if self._skip_multiline_comment():
                continue

            break

        if self._current_char is None:
            return Token(TokenType.EOF, None, self._location())

        # Handle newlines
        if self._current_char == "\n":
            loc = self._location()
            self._advance()
            return Token(TokenType.NEWLINE, "\n", loc)

        # F-string literals (f"..." or f'...')
        if self._current_char == "f" and self._peek_char in "\"'":
            return self._read_fstring()

        # String literals
        if self._current_char in "\"'":
            return self._read_string(self._current_char)

        # Numbers
        if self._current_char.isdigit():
            return self._read_number()

        # Mathematical symbols (Unicode) - check BEFORE identifiers
        # because some math symbols (like π) are considered alphabetic
        math_token = self._read_math_symbol()
        if math_token is not None:
            return math_token

        # Identifiers and keywords
        if self._current_char.isalpha() or self._current_char == "_":
            return self._read_identifier_or_keyword()

        # Operators and punctuation
        op_token = self._read_operator()
        if op_token is not None:
            return op_token

        # Unknown character
        raise LexerError(
            f"Unexpected character: {self._current_char!r}",
            self._location(),
            self._current_line_text(),
        )

    def tokenize(self) -> list[Token]:
        """
        Tokenize the entire source code.

        Returns:
            A list of all tokens including the final EOF token.
        """
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1
        self._line_start = 0

        while True:
            token = self._next_token()
            if token is None:
                break
            self.tokens.append(token)
            if token.type == TokenType.EOF:
                break

        return self.tokens

    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens (re-tokenizes if necessary)."""
        if not self.tokens:
            self.tokenize()
        return iter(self.tokens)


def tokenize(source: str, filename: Optional[str] = None) -> list[Token]:
    """
    Convenience function to tokenize source code.

    Args:
        source: MathViz source code
        filename: Optional filename for error reporting

    Returns:
        List of tokens
    """
    return Lexer(source, filename).tokenize()
