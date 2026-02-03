"""
MathViz Parser.

A recursive descent parser that transforms a token stream into an Abstract
Syntax Tree (AST). Implements operator precedence parsing for expressions
and handles all MathViz language constructs.
"""

from typing import Callable, Optional

from mathviz.compiler.tokens import Token, TokenType
from mathviz.compiler.ast_nodes import (
    # Program
    Program,
    # Types
    TypeAnnotation,
    SimpleType,
    GenericType,
    FunctionType,
    # Expressions
    Expression,
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BooleanLiteral,
    NoneLiteral,
    ListLiteral,
    SetLiteral,
    DictLiteral,
    TupleLiteral,
    SomeExpression,
    OkExpression,
    ErrExpression,
    UnwrapExpression,
    AwaitExpression,
    BinaryExpression,
    UnaryExpression,
    CallExpression,
    KeywordArgument,
    MemberAccess,
    IndexExpression,
    ConditionalExpression,
    LambdaExpression,
    RangeExpression,
    BinaryOperator,
    UnaryOperator,
    # Pattern matching
    Pattern,
    LiteralPattern,
    IdentifierPattern,
    TuplePattern,
    ConstructorPattern,
    RangePattern,
    OrPattern,
    BindingPattern,
    RestPattern,
    ListPattern,
    MatchArm,
    MatchExpression,
    # JIT/Numba
    JitMode,
    JitOptions,
    # Statements
    Statement,
    Block,
    ExpressionStatement,
    LetStatement,
    AssignmentStatement,
    CompoundAssignment,
    FunctionDef,
    AsyncFunctionDef,
    ClassDef,
    SceneDef,
    IfStatement,
    ForStatement,
    AsyncForStatement,
    WhileStatement,
    ReturnStatement,
    BreakStatement,
    ContinueStatement,
    PassStatement,
    ImportStatement,
    PrintStatement,
    UseStatement,
    ModuleDecl,
    PlayStatement,
    WaitStatement,
    Parameter,
    # OOP constructs
    Visibility,
    StructField,
    StructDef,
    Method,
    ImplBlock,
    AssociatedType,
    TraitMethod,
    TraitDef,
    EnumVariant,
    EnumDef,
    SelfExpression,
    EnumVariantAccess,
    StructLiteral,
    EnumPattern,
    # F-string
    FStringPart,
    FStringLiteral,
    FStringExpression,
    FString,
    # Const declarations
    ConstDeclaration,
    # Comprehensions and pipe lambdas
    ComprehensionClause,
    ListComprehension,
    SetComprehension,
    DictComprehension,
    PipeLambda,
    # Generic type parameters
    TypeParameter,
    WhereClause,
    # Documentation and attributes
    DocComment,
    Attribute,
)
from mathviz.utils.errors import ParserError, SourceLocation
from mathviz.utils.diagnostics import (
    DiagnosticEmitter,
    SourceSpan,
    Diagnostic,
    ErrorCode,
    create_unexpected_token_diagnostic,
    create_unclosed_delimiter_diagnostic,
)


# Operator precedence levels (higher = tighter binding)
class Precedence:
    """Operator precedence levels."""

    NONE = 0
    ASSIGNMENT = 1      # =
    CONDITIONAL = 2     # ? :
    OR = 3              # or
    AND = 4             # and
    EQUALITY = 5        # == !=
    COMPARISON = 6      # < > <= >=
    MEMBERSHIP = 7      # in, ∈, ⊆, etc.
    RANGE = 8           # ..
    BITWISE_OR = 9      # |
    BITWISE_XOR = 10    # ^^ (not exponentiation)
    BITWISE_AND = 11    # &
    SHIFT = 12          # << >>
    SET_OPS = 13        # ∪ ∩ ∖
    ADDITIVE = 14       # + -
    MULTIPLICATIVE = 15 # * / // %
    UNARY = 16          # - not
    POWER = 17          # ^ **
    POSTFIX = 18        # () [] .


# Map token types to binary operators
BINARY_OP_MAP: dict[TokenType, BinaryOperator] = {
    # Arithmetic
    TokenType.PLUS: BinaryOperator.ADD,
    TokenType.MINUS: BinaryOperator.SUB,
    TokenType.STAR: BinaryOperator.MUL,
    TokenType.SLASH: BinaryOperator.DIV,
    TokenType.DOUBLE_SLASH: BinaryOperator.FLOOR_DIV,
    TokenType.PERCENT: BinaryOperator.MOD,
    TokenType.CARET: BinaryOperator.POW,
    TokenType.DOUBLE_STAR: BinaryOperator.POW,
    # Comparison
    TokenType.EQ: BinaryOperator.EQ,
    TokenType.NE: BinaryOperator.NE,
    TokenType.NOT_EQUAL_MATH: BinaryOperator.NE,
    TokenType.LT: BinaryOperator.LT,
    TokenType.GT: BinaryOperator.GT,
    TokenType.LE: BinaryOperator.LE,
    TokenType.GE: BinaryOperator.GE,
    TokenType.LE_MATH: BinaryOperator.LE,
    TokenType.GE_MATH: BinaryOperator.GE,
    # Logical
    TokenType.AND: BinaryOperator.AND,
    TokenType.OR: BinaryOperator.OR,
    # Membership / Set
    TokenType.IN: BinaryOperator.IN,
    TokenType.ELEMENT_OF: BinaryOperator.ELEMENT_OF,
    TokenType.NOT_ELEMENT_OF: BinaryOperator.NOT_ELEMENT_OF,
    TokenType.SUBSET: BinaryOperator.SUBSET,
    TokenType.SUPERSET: BinaryOperator.SUPERSET,
    TokenType.PROPER_SUBSET: BinaryOperator.PROPER_SUBSET,
    TokenType.PROPER_SUPERSET: BinaryOperator.PROPER_SUPERSET,
    TokenType.UNION: BinaryOperator.UNION,
    TokenType.INTERSECTION: BinaryOperator.INTERSECTION,
    TokenType.SET_DIFF: BinaryOperator.SET_DIFF,
}

# Map token types to their precedence
PRECEDENCE_MAP: dict[TokenType, int] = {
    # NOTE: Assignment operators (=, +=, -=, *=, /=) are NOT in this map.
    # They are handled separately by _parse_expression_or_assignment() at the
    # statement level. Including them here causes infinite loops because
    # _parse_infix() doesn't handle them and returns without advancing.
    #
    # Logical
    TokenType.OR: Precedence.OR,
    TokenType.AND: Precedence.AND,
    # Equality
    TokenType.EQ: Precedence.EQUALITY,
    TokenType.NE: Precedence.EQUALITY,
    TokenType.NOT_EQUAL_MATH: Precedence.EQUALITY,
    # Comparison
    TokenType.LT: Precedence.COMPARISON,
    TokenType.GT: Precedence.COMPARISON,
    TokenType.LE: Precedence.COMPARISON,
    TokenType.GE: Precedence.COMPARISON,
    TokenType.LE_MATH: Precedence.COMPARISON,
    TokenType.GE_MATH: Precedence.COMPARISON,
    # Membership
    TokenType.IN: Precedence.MEMBERSHIP,
    TokenType.ELEMENT_OF: Precedence.MEMBERSHIP,
    TokenType.NOT_ELEMENT_OF: Precedence.MEMBERSHIP,
    TokenType.SUBSET: Precedence.MEMBERSHIP,
    TokenType.SUPERSET: Precedence.MEMBERSHIP,
    TokenType.PROPER_SUBSET: Precedence.MEMBERSHIP,
    TokenType.PROPER_SUPERSET: Precedence.MEMBERSHIP,
    # Set operations
    TokenType.UNION: Precedence.SET_OPS,
    TokenType.INTERSECTION: Precedence.SET_OPS,
    TokenType.SET_DIFF: Precedence.SET_OPS,
    # Additive
    TokenType.PLUS: Precedence.ADDITIVE,
    TokenType.MINUS: Precedence.ADDITIVE,
    # Multiplicative
    TokenType.STAR: Precedence.MULTIPLICATIVE,
    TokenType.SLASH: Precedence.MULTIPLICATIVE,
    TokenType.DOUBLE_SLASH: Precedence.MULTIPLICATIVE,
    TokenType.PERCENT: Precedence.MULTIPLICATIVE,
    # Power (right associative)
    TokenType.CARET: Precedence.POWER,
    TokenType.DOUBLE_STAR: Precedence.POWER,
    # Postfix
    TokenType.LPAREN: Precedence.POSTFIX,
    TokenType.LBRACKET: Precedence.POSTFIX,
    TokenType.DOT: Precedence.POSTFIX,
    TokenType.QUESTION: Precedence.POSTFIX,  # ? unwrap operator
}

# Keywords that are allowed as member names (after a DOT)
# This enables syntax like `circle.animate.scale(2)` where `animate` is a keyword
MEMBER_NAME_KEYWORDS: set[TokenType] = {
    TokenType.ANIMATE,
    TokenType.PLAY,
    TokenType.WAIT,
    TokenType.SELF,
}


class Parser:
    """
    Recursive descent parser for MathViz.

    Parses a list of tokens into an Abstract Syntax Tree.

    Usage:
        parser = Parser(tokens)
        ast = parser.parse()
    """

    def __init__(self, tokens: list[Token], source: str = "",
                 filename: str = "<input>") -> None:
        """
        Initialize the parser.

        Args:
            tokens: List of tokens from the lexer
            source: Optional source code for rich diagnostics
            filename: Optional filename for error reporting
        """
        self.tokens = tokens
        self.pos = 0
        self._source_lines: list[str] = source.splitlines() if source else []
        self._source = source
        self._filename = filename
        self._emitter: Optional[DiagnosticEmitter] = None
        self._delimiter_stack: list[tuple[str, SourceLocation]] = []  # Track open delimiters
        self.diagnostics: list[Diagnostic] = []

        # Initialize emitter if source provided
        if source:
            self._emitter = DiagnosticEmitter(source, filename)

    @property
    def _current(self) -> Token:
        """Get the current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    @property
    def _previous(self) -> Token:
        """Get the previous token."""
        return self.tokens[self.pos - 1] if self.pos > 0 else self.tokens[0]

    def _peek(self, offset: int = 1) -> Token:
        """Peek at a token ahead of the current position."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[pos]

    def _is_at_end(self) -> bool:
        """Check if we've reached the end of tokens."""
        return self._current.type == TokenType.EOF

    def _check(self, *types: TokenType) -> bool:
        """Check if the current token is one of the given types."""
        return self._current.type in types

    def _advance(self) -> Token:
        """Consume and return the current token."""
        token = self._current
        if not self._is_at_end():
            self.pos += 1
        return token

    def _match(self, *types: TokenType) -> bool:
        """Consume current token if it matches one of the given types."""
        if self._check(*types):
            self._advance()
            return True
        return False

    def _expect(self, token_type: TokenType, message: str) -> Token:
        """Consume current token if it matches, else raise error."""
        if self._check(token_type):
            return self._advance()
        raise self._error_with_context(message, expected=token_type.name)

    def _parse_member_name(self) -> str:
        """Parse a member name (identifier or allowed keyword).

        This allows certain keywords like 'animate' to be used as member names,
        enabling syntax like `circle.animate.scale(2)`.
        """
        if self._check(TokenType.IDENTIFIER):
            return self._advance().value

        if self._current.type in MEMBER_NAME_KEYWORDS:
            return self._advance().value

        raise self._error_with_context("Expected member name")

    def _error(self, message: str) -> ParserError:
        """Create a parser error with location info."""
        token = self._current
        return ParserError(message, token.location)

    def _error_with_context(self, message: str, expected: Optional[str] = None) -> ParserError:
        """Create a parser error with rich diagnostic context."""
        token = self._current

        # Create rich diagnostic if emitter is available
        if self._emitter and self._source:
            span = SourceSpan.from_location(
                token.location.line,
                token.location.column,
                len(str(token.value)) if token.value else 1,
                self._filename,
            )

            found = token.type.name if token.type != TokenType.EOF else "end of file"
            if expected:
                diagnostic = create_unexpected_token_diagnostic(
                    self._emitter,
                    expected,
                    found,
                    span,
                )
            else:
                diagnostic = self._emitter.error(
                    ErrorCode.E0201,
                    message,
                    span,
                ).emit()

            self.diagnostics.append(diagnostic)

        return ParserError(message, token.location)

    def _error_unclosed_delimiter(self, delimiter: str, open_loc: SourceLocation) -> ParserError:
        """Create an error for an unclosed delimiter with helpful context."""
        token = self._current

        if self._emitter and self._source:
            open_span = SourceSpan.from_location(
                open_loc.line,
                open_loc.column,
                1,
                self._filename,
            )
            error_span = SourceSpan.from_location(
                token.location.line,
                token.location.column,
                1,
                self._filename,
            )

            diagnostic = create_unclosed_delimiter_diagnostic(
                self._emitter,
                delimiter,
                open_span,
                error_span,
            )
            self.diagnostics.append(diagnostic)

        return ParserError(f"Unclosed delimiter '{delimiter}'", token.location)

    def get_diagnostics(self) -> list[Diagnostic]:
        """Get all rich diagnostics emitted during parsing."""
        return self.diagnostics

    def render_diagnostics(self, use_color: bool = True) -> str:
        """Render all diagnostics as formatted strings."""
        if self._emitter:
            return self._emitter.render_all(use_color)
        return ""

    def _skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self._match(TokenType.NEWLINE):
            pass

    def _synchronize(self) -> None:
        """
        Recover from a parse error by advancing to the next statement.

        Used for error recovery to continue parsing after an error.
        """
        self._advance()
        while not self._is_at_end():
            if self._previous.type == TokenType.NEWLINE:
                return
            if self._check(
                TokenType.LET,
                TokenType.FN,
                TokenType.CLASS,
                TokenType.SCENE,
                TokenType.IF,
                TokenType.FOR,
                TokenType.WHILE,
                TokenType.RETURN,
                TokenType.IMPORT,
            ):
                return
            self._advance()

    # -------------------------------------------------------------------------
    # Program Parsing
    # -------------------------------------------------------------------------

    def parse(self) -> Program:
        """
        Parse the entire program.

        Returns:
            The root Program AST node.
        """
        statements: list[Statement] = []
        self._skip_newlines()

        while not self._is_at_end():
            try:
                stmt = self._parse_statement()
                if stmt is not None:
                    statements.append(stmt)
            except ParserError as e:
                raise e  # Re-raise for now; could collect for error recovery
            self._skip_newlines()

        return Program(tuple(statements))

    # -------------------------------------------------------------------------
    # Statement Parsing
    # -------------------------------------------------------------------------

    def _parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        self._skip_newlines()

        # Collect doc comments to attach to the next declaration
        doc_comment = None
        if self._check(TokenType.DOC_COMMENT):
            doc_comment = self._advance().value
            self._skip_newlines()

        if self._check(TokenType.IMPORT, TokenType.FROM):
            return self._parse_import()
        if self._check(TokenType.USE):
            return self._parse_use()
        if self._check(TokenType.MOD):
            return self._parse_module()
        if self._check(TokenType.LET):
            return self._parse_let()
        if self._check(TokenType.CONST):
            return self._parse_const()
        if self._check(TokenType.AT):
            return self._parse_decorated_function()
        if self._check(TokenType.ASYNC):
            return self._parse_async_statement()
        if self._check(TokenType.FN):
            return self._parse_function_def()
        if self._check(TokenType.PUB):
            return self._parse_pub_declaration()
        if self._check(TokenType.CLASS):
            return self._parse_class_def()
        if self._check(TokenType.SCENE):
            return self._parse_scene_def()
        if self._check(TokenType.IF):
            return self._parse_if()
        if self._check(TokenType.FOR):
            return self._parse_for()
        if self._check(TokenType.WHILE):
            return self._parse_while()
        if self._check(TokenType.RETURN):
            return self._parse_return()
        if self._check(TokenType.BREAK):
            return self._parse_break()
        if self._check(TokenType.CONTINUE):
            return self._parse_continue()
        if self._check(TokenType.PASS):
            return self._parse_pass()
        if self._check(TokenType.PRINT):
            return self._parse_print(newline=False)
        if self._check(TokenType.PRINTLN):
            return self._parse_print(newline=True)
        if self._check(TokenType.PLAY):
            return self._parse_play()
        if self._check(TokenType.WAIT):
            return self._parse_wait()
        # OOP constructs
        if self._check(TokenType.STRUCT):
            return self._parse_struct_def()
        if self._check(TokenType.IMPL):
            return self._parse_impl_block()
        if self._check(TokenType.TRAIT):
            return self._parse_trait_def()
        if self._check(TokenType.ENUM):
            return self._parse_enum_def()

        # Expression statement (or assignment)
        return self._parse_expression_or_assignment()

    def _parse_import(self) -> ImportStatement:
        """
        Parse an import statement.

        Handles:
            import module
            import module as alias
            from module import name1, name2
            from module import name1 as alias1, name2 as alias2
        """
        loc = self._current.location

        if self._match(TokenType.FROM):
            # from ... import ...
            module = self._expect(TokenType.IDENTIFIER, "Expected module name").value
            self._expect(TokenType.IMPORT, "Expected 'import' after module name")

            names: list[tuple[str, Optional[str]]] = []
            while True:
                name = self._expect(TokenType.IDENTIFIER, "Expected import name").value
                alias = None
                if self._match(TokenType.AS):
                    alias = self._expect(
                        TokenType.IDENTIFIER, "Expected alias name"
                    ).value
                names.append((name, alias))
                if not self._match(TokenType.COMMA):
                    break

            return ImportStatement(
                module=module,
                names=tuple(names),
                is_from_import=True,
                location=loc,
            )
        else:
            # import ...
            self._advance()  # consume 'import'
            module = self._expect(TokenType.IDENTIFIER, "Expected module name").value
            alias = None
            if self._match(TokenType.AS):
                alias = self._expect(TokenType.IDENTIFIER, "Expected alias name").value

            return ImportStatement(module=module, alias=alias, location=loc)

    def _parse_let(self) -> LetStatement:
        """
        Parse a let (variable declaration) statement.

        Handles:
            let x = value
            let x: Type = value
            let x: Type
            let mut x = value  (mutable variable)
        """
        loc = self._current.location
        self._advance()  # consume 'let'

        # Check for mut keyword
        mutable = self._match(TokenType.MUT)

        name = self._expect(TokenType.IDENTIFIER, "Expected variable name").value

        type_annotation: Optional[TypeAnnotation] = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type_annotation()

        value: Optional[Expression] = None
        if self._match(TokenType.ASSIGN):
            value = self._parse_expression()

        return LetStatement(
            name=name,
            type_annotation=type_annotation,
            value=value,
            mutable=mutable,
            location=loc,
        )

    def _parse_const(self) -> ConstDeclaration:
        """
        Parse a const (compile-time constant) declaration.

        Handles:
            const PI = 3.14159
            const MAX_SIZE: Int = 1024
            const VERSION = "1.0.0"

        Constants MUST have an initializer.
        """
        loc = self._current.location
        self._advance()  # consume 'const'

        name = self._expect(TokenType.IDENTIFIER, "Expected constant name").value

        type_annotation: Optional[TypeAnnotation] = None
        if self._match(TokenType.COLON):
            type_annotation = self._parse_type_annotation()

        # Constants must have an initializer
        self._expect(TokenType.ASSIGN, "Expected '=' in const declaration (constants must have a value)")
        value = self._parse_expression()

        return ConstDeclaration(
            name=name,
            value=value,
            type_annotation=type_annotation,
            location=loc,
        )

    def _parse_decorated_function(self) -> FunctionDef:
        """
        Parse a function with JIT decorators.

        Handles:
            @jit
            fn name(params) { body }

            @njit(parallel=true, fastmath=true)
            fn compute(arr: List[Float]) -> Float { ... }
        """
        jit_options = self._parse_jit_decorator()
        self._skip_newlines()

        if not self._check(TokenType.FN):
            raise self._error("Expected 'fn' after decorator")

        return self._parse_function_def(jit_options)

    def _parse_jit_decorator(self) -> JitOptions:
        """
        Parse a JIT decorator with optional options.

        Handles:
            @jit
            @njit
            @jit(parallel=true, cache=true)
            @njit(fastmath=true, nogil=true)
        """
        self._advance()  # consume '@'

        # Determine JIT mode
        if self._match(TokenType.JIT):
            mode = JitMode.JIT
        elif self._match(TokenType.NJIT):
            mode = JitMode.NJIT
        elif self._match(TokenType.VECTORIZE):
            mode = JitMode.VECTORIZE
        else:
            raise self._error("Expected 'jit', 'njit', or 'vectorize' after '@'")

        # Default options
        nopython = True
        nogil = False
        cache = True
        parallel = False
        fastmath = False
        boundscheck = False
        inline = "never"

        # Parse optional arguments
        if self._match(TokenType.LPAREN):
            while not self._check(TokenType.RPAREN):
                # Option name can be an identifier or a keyword like 'parallel'
                if self._check(TokenType.IDENTIFIER):
                    opt_name = self._advance().value
                elif self._check(TokenType.PARALLEL):
                    opt_name = "parallel"
                    self._advance()
                elif self._check(TokenType.IN):
                    opt_name = "inline"
                    self._advance()
                else:
                    # Try to get any token as option name
                    opt_name = self._current.value
                    if isinstance(opt_name, str):
                        self._advance()
                    else:
                        raise self._error("Expected option name")

                self._expect(TokenType.ASSIGN, "Expected '=' after option name")

                # Parse boolean or string value
                if self._match(TokenType.TRUE):
                    opt_value = True
                elif self._match(TokenType.FALSE):
                    opt_value = False
                elif self._check(TokenType.STRING):
                    opt_value = self._advance().value
                elif self._check(TokenType.IDENTIFIER):
                    # Handle 'true'/'false' as identifiers too
                    val = self._advance().value
                    if val == "true":
                        opt_value = True
                    elif val == "false":
                        opt_value = False
                    else:
                        opt_value = val
                else:
                    raise self._error("Expected boolean or string value")

                # Apply option
                if opt_name == "nopython":
                    nopython = bool(opt_value)
                elif opt_name == "nogil":
                    nogil = bool(opt_value)
                elif opt_name == "cache":
                    cache = bool(opt_value)
                elif opt_name == "parallel":
                    parallel = bool(opt_value)
                elif opt_name == "fastmath":
                    fastmath = bool(opt_value)
                elif opt_name == "boundscheck":
                    boundscheck = bool(opt_value)
                elif opt_name == "inline":
                    inline = str(opt_value)
                else:
                    raise self._error(f"Unknown JIT option: {opt_name}")

                if not self._match(TokenType.COMMA):
                    break

            self._expect(TokenType.RPAREN, "Expected ')' after decorator options")

        return JitOptions(
            mode=mode,
            nopython=nopython,
            nogil=nogil,
            cache=cache,
            parallel=parallel,
            fastmath=fastmath,
            boundscheck=boundscheck,
            inline=inline,
        )

    def _parse_function_def(self, jit_options: Optional[JitOptions] = None) -> FunctionDef:
        """
        Parse a function definition, with optional generics.

        Handles:
            fn name(params) { body }
            fn name(params) -> ReturnType { body }
            fn name(params) -> ReturnType = expr  (expression body)
            fn name<T, U>(params) -> ReturnType { body }
            fn name<T: Bound>(params) -> ReturnType { body }
            fn name<T>(params) -> ReturnType where T: Bound { body }
        """
        loc = self._current.location
        self._advance()  # consume 'fn'

        name = self._expect(TokenType.IDENTIFIER, "Expected function name").value

        # Optional type parameters <T, U: Bound>
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Parameters
        self._expect(TokenType.LPAREN, "Expected '(' after function name")
        params = self._parse_parameters()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")

        # Optional return type
        return_type: Optional[TypeAnnotation] = None
        if self._match(TokenType.THIN_ARROW):
            return_type = self._parse_type_annotation()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        # Body: either block { } or expression body = expr
        self._skip_newlines()
        if self._match(TokenType.ASSIGN):
            # Expression body: fn name(params) -> Type = expr
            expr = self._parse_expression()
            # Wrap expression in a return statement inside a block
            return_stmt = ReturnStatement(value=expr, location=expr.location)
            body = Block(statements=(return_stmt,), location=loc)
        else:
            # Block body: fn name(params) { body }
            body = self._parse_block()

        return FunctionDef(
            name=name,
            parameters=tuple(params),
            return_type=return_type,
            body=body,
            type_params=type_params,
            where_clause=where_clause,
            jit_options=jit_options or JitOptions(),
            location=loc,
        )

    def _parse_parameters(self) -> list[Parameter]:
        """Parse function parameters."""
        params: list[Parameter] = []

        if self._check(TokenType.RPAREN):
            return params

        while True:
            loc = self._current.location

            # Accept 'self' as a valid parameter name (for class methods)
            if self._match(TokenType.SELF):
                name = "self"
            else:
                name = self._expect(TokenType.IDENTIFIER, "Expected parameter name").value

            type_annotation: Optional[TypeAnnotation] = None
            if self._match(TokenType.COLON):
                type_annotation = self._parse_type_annotation()

            default_value: Optional[Expression] = None
            if self._match(TokenType.ASSIGN):
                default_value = self._parse_expression()

            params.append(
                Parameter(
                    name=name,
                    type_annotation=type_annotation,
                    default_value=default_value,
                    location=loc,
                )
            )

            if not self._match(TokenType.COMMA):
                break

        return params

    def _parse_type_parameters(self) -> tuple[TypeParameter, ...]:
        """
        Parse type parameters for generics.

        Syntax:
            <T>
            <T, U>
            <T: Bound>
            <T: Bound1 + Bound2>
            <T: Bound, U: OtherBound>
        """
        type_params: list[TypeParameter] = []

        self._expect(TokenType.LT, "Expected '<' to start type parameters")

        while True:
            loc = self._current.location
            name = self._expect(TokenType.IDENTIFIER, "Expected type parameter name").value

            # Optional bounds: T: Bound1 + Bound2
            bounds: list[str] = []
            if self._match(TokenType.COLON):
                bounds = self._parse_type_bounds()

            type_params.append(TypeParameter(name=name, bounds=tuple(bounds), location=loc))

            if not self._match(TokenType.COMMA):
                break

        self._expect(TokenType.GT, "Expected '>' to close type parameters")

        return tuple(type_params)

    def _parse_type_bounds(self) -> list[str]:
        """
        Parse trait bounds for a type parameter.

        Syntax:
            Display
            Display + Clone
            Display + Clone + Debug
        """
        bounds: list[str] = []

        # First bound
        bound_name = self._expect(TokenType.IDENTIFIER, "Expected trait bound name").value
        bounds.append(bound_name)

        # Additional bounds: + Bound2 + Bound3
        while self._match(TokenType.PLUS):
            bound_name = self._expect(TokenType.IDENTIFIER, "Expected trait bound name after '+'").value
            bounds.append(bound_name)

        return bounds

    def _parse_where_clause(self) -> WhereClause:
        """
        Parse a where clause for complex type constraints.

        Syntax:
            where T: Display
            where T: Display, U: Clone
            where T: Display + Clone, U: Debug
        """
        loc = self._current.location
        self._advance()  # consume 'where'

        constraints: list[tuple[str, tuple[str, ...]]] = []

        while True:
            type_param = self._expect(TokenType.IDENTIFIER, "Expected type parameter name in where clause").value
            self._expect(TokenType.COLON, "Expected ':' after type parameter in where clause")
            bounds = self._parse_type_bounds()
            constraints.append((type_param, tuple(bounds)))

            if not self._match(TokenType.COMMA):
                break

        return WhereClause(constraints=tuple(constraints), location=loc)

    def _parse_class_def(self) -> ClassDef:
        """
        Parse a class definition.

        Handles:
            class Name { body }
            class Name(Base1, Base2) { body }
        """
        loc = self._current.location
        self._advance()  # consume 'class'

        name = self._expect(TokenType.IDENTIFIER, "Expected class name").value

        # Optional base classes
        base_classes: list[str] = []
        if self._match(TokenType.LPAREN):
            while True:
                base = self._expect(
                    TokenType.IDENTIFIER, "Expected base class name"
                ).value
                base_classes.append(base)
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN, "Expected ')' after base classes")

        self._skip_newlines()
        body = self._parse_block()

        return ClassDef(
            name=name,
            base_classes=tuple(base_classes),
            body=body,
            location=loc,
        )

    def _parse_scene_def(self) -> SceneDef:
        """
        Parse a Manim scene definition.

        Handles:
            scene Name { body }
            scene Name extends Scene { body }
        """
        loc = self._current.location
        self._advance()  # consume 'scene'

        name = self._expect(TokenType.IDENTIFIER, "Expected scene name").value

        # Optional 'extends' clause (ignored - all scenes extend Scene)
        if self._check(TokenType.IDENTIFIER) and self._current.value == "extends":
            self._advance()  # consume 'extends'
            self._expect(TokenType.IDENTIFIER, "Expected base class name")  # consume 'Scene'

        self._skip_newlines()
        body = self._parse_block()

        return SceneDef(name=name, body=body, location=loc)

    def _parse_if(self) -> IfStatement:
        """
        Parse an if statement with optional elif and else clauses.

        Handles:
            if cond { body }
            if cond { body } else { body }
            if cond { body } elif cond { body } else { body }
        """
        loc = self._current.location
        self._advance()  # consume 'if'

        condition = self._parse_expression()
        self._skip_newlines()
        then_block = self._parse_block()

        elif_clauses: list[tuple[Expression, Block]] = []
        else_block: Optional[Block] = None

        self._skip_newlines()
        while self._match(TokenType.ELIF):
            elif_cond = self._parse_expression()
            self._skip_newlines()
            elif_block = self._parse_block()
            elif_clauses.append((elif_cond, elif_block))
            self._skip_newlines()

        if self._match(TokenType.ELSE):
            self._skip_newlines()
            else_block = self._parse_block()

        return IfStatement(
            condition=condition,
            then_block=then_block,
            elif_clauses=tuple(elif_clauses),
            else_block=else_block,
            location=loc,
        )

    def _parse_for(self) -> ForStatement:
        """
        Parse a for loop.

        Handles:
            for var in iterable { body }
        """
        loc = self._current.location
        self._advance()  # consume 'for'

        variable = self._expect(TokenType.IDENTIFIER, "Expected loop variable").value

        self._expect(TokenType.IN, "Expected 'in' after loop variable")

        iterable = self._parse_expression()
        self._skip_newlines()
        body = self._parse_block()

        return ForStatement(
            variable=variable,
            iterable=iterable,
            body=body,
            location=loc,
        )

    def _parse_async_statement(self) -> Statement:
        """
        Parse an async statement.

        Handles:
            async fn name(params) -> ReturnType { body }
            async for var in stream { body }
        """
        loc = self._current.location
        self._advance()  # consume 'async'
        self._skip_newlines()

        if self._check(TokenType.FN):
            return self._parse_async_function_def(loc)
        elif self._check(TokenType.FOR):
            return self._parse_async_for(loc)
        else:
            raise self._error("Expected 'fn' or 'for' after 'async'")

    def _parse_async_function_def(self, loc: SourceLocation) -> AsyncFunctionDef:
        """
        Parse an async function definition.

        Handles:
            async fn name(params) { body }
            async fn name(params) -> ReturnType { body }
            async fn name<T>(params) -> ReturnType { body }
        """
        self._advance()  # consume 'fn'

        name = self._expect(TokenType.IDENTIFIER, "Expected function name").value

        # Optional type parameters <T, U: Bound>
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Parameters
        self._expect(TokenType.LPAREN, "Expected '(' after function name")
        params = self._parse_parameters()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")

        # Optional return type
        return_type: Optional[TypeAnnotation] = None
        if self._match(TokenType.THIN_ARROW):
            return_type = self._parse_type_annotation()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        # Body
        self._skip_newlines()
        body = self._parse_block()

        return AsyncFunctionDef(
            name=name,
            parameters=tuple(params),
            return_type=return_type,
            body=body,
            type_params=type_params,
            where_clause=where_clause,
            location=loc,
        )

    def _parse_async_for(self, loc: SourceLocation) -> AsyncForStatement:
        """
        Parse an async for loop.

        Handles:
            async for var in stream { body }
        """
        self._advance()  # consume 'for'

        variable = self._expect(TokenType.IDENTIFIER, "Expected loop variable").value

        self._expect(TokenType.IN, "Expected 'in' after loop variable")

        iterable = self._parse_expression()
        self._skip_newlines()
        body = self._parse_block()

        return AsyncForStatement(
            variable=variable,
            iterable=iterable,
            body=body,
            location=loc,
        )

    def _parse_while(self) -> WhileStatement:
        """
        Parse a while loop.

        Handles:
            while condition { body }
        """
        loc = self._current.location
        self._advance()  # consume 'while'

        condition = self._parse_expression()
        self._skip_newlines()
        body = self._parse_block()

        return WhileStatement(
            condition=condition,
            body=body,
            location=loc,
        )

    def _parse_return(self) -> ReturnStatement:
        """Parse a return statement."""
        loc = self._current.location
        self._advance()  # consume 'return'

        value: Optional[Expression] = None
        if not self._check(TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF):
            value = self._parse_expression()

        return ReturnStatement(value=value, location=loc)

    def _parse_break(self) -> BreakStatement:
        """Parse a break statement."""
        loc = self._current.location
        self._advance()  # consume 'break'
        return BreakStatement(location=loc)

    def _parse_continue(self) -> ContinueStatement:
        """Parse a continue statement."""
        loc = self._current.location
        self._advance()  # consume 'continue'
        return ContinueStatement(location=loc)

    def _parse_pass(self) -> PassStatement:
        """Parse a pass statement."""
        loc = self._current.location
        self._advance()  # consume 'pass'
        return PassStatement(location=loc)

    def _parse_print(self, newline: bool) -> PrintStatement:
        """
        Parse a print or println statement.

        Handles:
            print("Hello")
            println("Value: {}", x)
            println("x={}, y={}", x, y)
        """
        loc = self._current.location
        self._advance()  # consume 'print' or 'println'

        self._expect(TokenType.LPAREN, "Expected '(' after print")

        # Parse format string
        format_string = self._parse_expression()

        # Parse optional arguments
        arguments: list[Expression] = []
        while self._match(TokenType.COMMA):
            arguments.append(self._parse_expression())

        self._expect(TokenType.RPAREN, "Expected ')' after print arguments")

        return PrintStatement(
            format_string=format_string,
            arguments=tuple(arguments),
            newline=newline,
            location=loc,
        )

    def _parse_use(self) -> UseStatement:
        """
        Parse a use statement.

        Handles:
            use manim.*
            use mylib.topology
            use std.io as io
        """
        loc = self._current.location
        self._advance()  # consume 'use'

        # Parse module path
        path_parts: list[str] = []
        path_parts.append(self._expect(TokenType.IDENTIFIER, "Expected module name").value)

        wildcard = False
        while self._match(TokenType.DOT):
            if self._match(TokenType.STAR):
                wildcard = True
                break
            path_parts.append(self._expect(TokenType.IDENTIFIER, "Expected module name").value)

        # Optional alias
        alias = None
        if self._match(TokenType.AS):
            alias = self._expect(TokenType.IDENTIFIER, "Expected alias name").value

        return UseStatement(
            module_path=tuple(path_parts),
            wildcard=wildcard,
            alias=alias,
            location=loc,
        )

    def _parse_module(self) -> ModuleDecl:
        """
        Parse a module declaration.

        Handles:
            mod topology { ... }
        """
        loc = self._current.location
        self._advance()  # consume 'mod'

        name = self._expect(TokenType.IDENTIFIER, "Expected module name").value

        self._skip_newlines()
        body = self._parse_block()

        return ModuleDecl(
            name=name,
            body=body,
            is_public=False,
            location=loc,
        )

    def _parse_pub_declaration(self) -> Statement:
        """
        Parse a public declaration.

        Handles:
            pub fn foo() { ... }
            pub let x = 10
            pub struct Point { ... }
            pub trait Shape { ... }
            pub enum Color { ... }
        """
        self._advance()  # consume 'pub'
        self._skip_newlines()

        # Parse the actual declaration
        if self._check(TokenType.FN):
            func = self._parse_function_def()
            # Mark as public (would need to add this to FunctionDef)
            return func
        elif self._check(TokenType.MOD):
            mod = self._parse_module()
            return ModuleDecl(
                name=mod.name,
                body=mod.body,
                is_public=True,
                location=mod.location,
            )
        elif self._check(TokenType.STRUCT):
            return self._parse_struct_def()
        elif self._check(TokenType.TRAIT):
            return self._parse_trait_def()
        elif self._check(TokenType.ENUM):
            return self._parse_enum_def()

        raise self._error("Expected 'fn', 'mod', 'struct', 'trait', or 'enum' after 'pub'")

    def _parse_play(self) -> PlayStatement:
        """
        Parse a Manim play statement (supports multiline and multiple animations).

        Handles:
            play(Create(circle))
            play(FadeIn(a), Write(b))
            play(Transform(a, b), run_time=2.0)
            play(
                circle.animate.set_color(PURPLE),
                square.animate.set_color(ORANGE)
            )
        """
        loc = self._current.location
        self._advance()  # consume 'play'

        self._expect(TokenType.LPAREN, "Expected '(' after play")

        # Skip newlines after opening paren
        self._skip_newlines()

        # Parse first animation
        animations: list[Expression] = [self._parse_expression()]

        self._skip_newlines()

        # Check for more animations or run_time keyword argument
        run_time = None
        while self._match(TokenType.COMMA):
            self._skip_newlines()
            # Check if it's run_time=value
            if self._check(TokenType.IDENTIFIER) and self._current.value == "run_time":
                self._advance()
                self._expect(TokenType.ASSIGN, "Expected '=' after run_time")
                run_time = self._parse_expression()
                break
            # Otherwise it's another animation
            animations.append(self._parse_expression())
            self._skip_newlines()

        self._skip_newlines()
        self._expect(TokenType.RPAREN, "Expected ')' after play arguments")

        # If multiple animations, wrap in a tuple for codegen
        if len(animations) == 1:
            animation = animations[0]
        else:
            # Create a tuple of animations
            animation = TupleLiteral(elements=tuple(animations), location=loc)

        return PlayStatement(
            animation=animation,
            run_time=run_time,
            location=loc,
        )

    def _parse_wait(self) -> WaitStatement:
        """
        Parse a Manim wait statement.

        Handles:
            wait()
            wait(1.0)
        """
        loc = self._current.location
        self._advance()  # consume 'wait'

        duration = None
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                duration = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after wait duration")

        return WaitStatement(
            duration=duration,
            location=loc,
        )

    def _parse_block(self) -> Block:
        """
        Parse a block of statements enclosed in braces.

        Handles:
            { stmt1; stmt2; ... }
        """
        loc = self._current.location
        open_loc = self._current.location

        if not self._check(TokenType.LBRACE):
            raise self._error_with_context("Expected '{' to start block", expected="'{'")
        self._advance()  # consume '{'

        # Track the opening brace for better error messages
        self._delimiter_stack.append(('{', open_loc))
        self._skip_newlines()

        statements: list[Statement] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self._skip_newlines()
            # Allow optional semicolons between statements
            while self._match(TokenType.SEMICOLON):
                self._skip_newlines()

        # Check for unclosed delimiter
        if self._check(TokenType.EOF):
            self._delimiter_stack.pop()
            raise self._error_unclosed_delimiter('{', open_loc)

        self._expect(TokenType.RBRACE, "Expected '}' to end block")
        self._delimiter_stack.pop()

        return Block(tuple(statements), location=loc)

    def _parse_expression_or_assignment(self) -> Statement:
        """Parse an expression statement, handling assignments."""
        expr = self._parse_expression()

        # Check for assignment
        if self._match(TokenType.ASSIGN):
            value = self._parse_expression()
            return AssignmentStatement(
                target=expr,
                value=value,
                location=expr.location,
            )

        # Check for compound assignment
        compound_ops = {
            TokenType.PLUS_ASSIGN: BinaryOperator.ADD,
            TokenType.MINUS_ASSIGN: BinaryOperator.SUB,
            TokenType.STAR_ASSIGN: BinaryOperator.MUL,
            TokenType.SLASH_ASSIGN: BinaryOperator.DIV,
        }
        for token_type, op in compound_ops.items():
            if self._match(token_type):
                value = self._parse_expression()
                return CompoundAssignment(
                    target=expr,
                    operator=op,
                    value=value,
                    location=expr.location,
                )

        return ExpressionStatement(expr, location=expr.location)

    # -------------------------------------------------------------------------
    # Type Annotation Parsing
    # -------------------------------------------------------------------------

    def _parse_type_annotation(self) -> TypeAnnotation:
        """
        Parse a type annotation.

        Handles:
            Int, Float, Bool, String
            List[T], Set[T], Dict[K, V]
            (T1, T2) -> R
        """
        loc = self._current.location

        # Function type: (params) -> return
        if self._match(TokenType.LPAREN):
            param_types: list[TypeAnnotation] = []
            if not self._check(TokenType.RPAREN):
                while True:
                    param_types.append(self._parse_type_annotation())
                    if not self._match(TokenType.COMMA):
                        break
            self._expect(TokenType.RPAREN, "Expected ')' in function type")
            self._expect(TokenType.THIN_ARROW, "Expected '->' in function type")
            return_type = self._parse_type_annotation()
            return FunctionType(
                param_types=tuple(param_types),
                return_type=return_type,
                location=loc,
            )

        # Handle built-in type keywords
        type_keywords = {
            TokenType.TYPE_INT,
            TokenType.TYPE_FLOAT,
            TokenType.TYPE_BOOL,
            TokenType.TYPE_STRING,
            TokenType.TYPE_LIST,
            TokenType.TYPE_SET,
            TokenType.TYPE_DICT,
            TokenType.TYPE_TUPLE,
            TokenType.TYPE_OPTIONAL,
            TokenType.TYPE_RESULT,
        }

        # Simple or generic type - can be identifier OR type keyword
        if self._check(*type_keywords):
            token = self._advance()
            name = token.value
        elif self._check(TokenType.IDENTIFIER):
            name = self._advance().value
        else:
            raise self._error("Expected type name")

        # Check for generic type arguments
        if self._match(TokenType.LBRACKET):
            type_args: list[TypeAnnotation] = []
            while True:
                type_args.append(self._parse_type_annotation())
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.RBRACKET, "Expected ']' after type arguments")
            return GenericType(base=name, type_args=tuple(type_args), location=loc)

        return SimpleType(name=name, location=loc)

    # -------------------------------------------------------------------------
    # Expression Parsing (Pratt Parser / Precedence Climbing)
    # -------------------------------------------------------------------------

    def _parse_expression(self, min_precedence: int = Precedence.NONE) -> Expression:
        """
        Parse an expression using precedence climbing.

        This is the core of the Pratt parser, handling operator precedence
        correctly for all binary and unary operators.
        """
        left = self._parse_prefix()

        while True:
            precedence = PRECEDENCE_MAP.get(self._current.type, Precedence.NONE)
            if precedence <= min_precedence:
                break

            # Handle right-associative operators (exponentiation)
            if self._current.type in (TokenType.CARET, TokenType.DOUBLE_STAR):
                operator = self._advance()
                right = self._parse_expression(precedence - 1)  # Right assoc
                left = BinaryExpression(
                    left=left,
                    operator=BINARY_OP_MAP[operator.type],
                    right=right,
                    location=left.location,
                )
            else:
                left = self._parse_infix(left, precedence)

        return left

    def _parse_prefix(self) -> Expression:
        """Parse a prefix expression (unary operators, literals, etc.)."""
        loc = self._current.location

        # Unary minus
        if self._match(TokenType.MINUS):
            operand = self._parse_prefix()
            return UnaryExpression(
                operator=UnaryOperator.NEG,
                operand=operand,
                location=loc,
            )

        # Unary plus
        if self._match(TokenType.PLUS):
            operand = self._parse_prefix()
            return UnaryExpression(
                operator=UnaryOperator.POS,
                operand=operand,
                location=loc,
            )

        # Logical not
        if self._match(TokenType.NOT):
            operand = self._parse_prefix()
            return UnaryExpression(
                operator=UnaryOperator.NOT,
                operand=operand,
                location=loc,
            )

        # Await expression
        if self._match(TokenType.AWAIT):
            expr = self._parse_prefix()
            return AwaitExpression(
                expression=expr,
                location=loc,
            )

        # Parenthesized expression, tuple, or lambda
        if self._match(TokenType.LPAREN):
            return self._parse_grouped_or_tuple(loc)

        return self._parse_primary()

    def _parse_grouped_or_tuple(self, loc: SourceLocation) -> Expression:
        """
        Parse a parenthesized expression, tuple, or lambda.

        Handles:
            ()              -> Empty tuple
            (expr)          -> Grouped expression
            (expr,)         -> Single-element tuple
            (expr, expr, ...) -> Multi-element tuple
            () => body      -> Lambda with no params
            (x) => body     -> Lambda with one param
            (x, y) => body  -> Lambda with multiple params
        """
        # Empty parens: could be empty tuple or lambda with no params
        if self._check(TokenType.RPAREN):
            self._advance()
            if self._match(TokenType.FAT_ARROW):
                body = self._parse_lambda_body()
                return LambdaExpression(parameters=(), body=body, location=loc)
            # Empty tuple
            return TupleLiteral(elements=(), location=loc)

        # Parse first expression
        first_expr = self._parse_expression()

        # Check for comma -> tuple or lambda with multiple params
        if self._match(TokenType.COMMA):
            # Check if it's a trailing comma for single-element tuple
            if self._check(TokenType.RPAREN):
                self._advance()  # consume ')'
                # Check for lambda arrow
                if self._match(TokenType.FAT_ARROW):
                    param = self._expression_to_parameter(first_expr)
                    body = self._parse_lambda_body()
                    return LambdaExpression(
                        parameters=(param,),
                        body=body,
                        location=loc,
                    )
                # Single-element tuple: (expr,)
                return TupleLiteral(elements=(first_expr,), location=loc)

            # Multiple elements
            elements = [first_expr]
            while True:
                elements.append(self._parse_expression())
                if not self._match(TokenType.COMMA):
                    break
                # Check for trailing comma
                if self._check(TokenType.RPAREN):
                    break

            self._expect(TokenType.RPAREN, "Expected ')' after tuple/lambda elements")

            # Check for lambda arrow
            if self._match(TokenType.FAT_ARROW):
                params = [self._expression_to_parameter(e) for e in elements]
                body = self._parse_lambda_body()
                return LambdaExpression(
                    parameters=tuple(params),
                    body=body,
                    location=loc,
                )

            # Multi-element tuple
            return TupleLiteral(elements=tuple(elements), location=loc)

        # Single expression in parens
        self._expect(TokenType.RPAREN, "Expected ')' after expression")

        # Check if this is a single-param lambda
        if self._match(TokenType.FAT_ARROW):
            param = self._expression_to_parameter(first_expr)
            body = self._parse_lambda_body()
            return LambdaExpression(
                parameters=(param,),
                body=body,
                location=loc,
            )

        # Just a grouped expression (not a tuple)
        return first_expr

    def _parse_infix(self, left: Expression, precedence: int) -> Expression:
        """Parse an infix (binary) expression."""
        token = self._current
        loc = left.location

        # Function call
        if self._match(TokenType.LPAREN):
            positional_args, keyword_args = self._parse_arguments()
            self._expect(TokenType.RPAREN, "Expected ')' after arguments")
            left = CallExpression(
                callee=left,
                arguments=tuple(positional_args),
                keyword_arguments=tuple(keyword_args),
                location=loc
            )
            return self._continue_postfix(left)

        # Index access
        if self._match(TokenType.LBRACKET):
            index = self._parse_expression()
            self._expect(TokenType.RBRACKET, "Expected ']' after index")
            left = IndexExpression(object=left, index=index, location=loc)
            return self._continue_postfix(left)

        # Range expression (..) - check this BEFORE member access
        # Look ahead to see if we have two consecutive dots
        if self._check(TokenType.DOT):
            # Save position to potentially backtrack
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.DOT:
                # It's a range expression (..)
                self._advance()  # consume first dot
                self._advance()  # consume second dot
                inclusive = self._match(TokenType.ASSIGN)  # ..=
                end = self._parse_expression(Precedence.RANGE)
                return RangeExpression(
                    start=left,
                    end=end,
                    inclusive=inclusive,
                    location=loc,
                )

        # Member access (single dot)
        if self._match(TokenType.DOT):
            member = self._parse_member_name()
            left = MemberAccess(object=left, member=member, location=loc)
            return self._continue_postfix(left)

        # Unwrap operator (?)
        if self._match(TokenType.QUESTION):
            left = UnwrapExpression(operand=left, location=loc)
            return self._continue_postfix(left)

        # Binary operators
        if token.type in BINARY_OP_MAP:
            self._advance()
            right = self._parse_expression(precedence)
            return BinaryExpression(
                left=left,
                operator=BINARY_OP_MAP[token.type],
                right=right,
                location=loc,
            )

        return left

    def _continue_postfix(self, expr: Expression) -> Expression:
        """Continue parsing postfix operations (calls, indexing, member access, unwrap)."""
        while True:
            if self._match(TokenType.LPAREN):
                positional_args, keyword_args = self._parse_arguments()
                self._expect(TokenType.RPAREN, "Expected ')' after arguments")
                expr = CallExpression(
                    callee=expr,
                    arguments=tuple(positional_args),
                    keyword_arguments=tuple(keyword_args),
                    location=expr.location,
                )
            elif self._match(TokenType.LBRACKET):
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET, "Expected ']' after index")
                expr = IndexExpression(
                    object=expr,
                    index=index,
                    location=expr.location,
                )
            elif self._match(TokenType.DOT):
                member = self._parse_member_name()
                expr = MemberAccess(
                    object=expr,
                    member=member,
                    location=expr.location,
                )
            elif self._match(TokenType.QUESTION):
                # Unwrap operator (?)
                expr = UnwrapExpression(
                    operand=expr,
                    location=expr.location,
                )
            else:
                break
        return expr

    def _parse_primary(self) -> Expression:
        """Parse a primary expression (literals, identifiers, etc.)."""
        loc = self._current.location

        # Pipe lambda: |x| expr or |x, y| expr
        if self._match(TokenType.PIPE):
            return self._parse_pipe_lambda(loc)

        # Integer literal
        if self._match(TokenType.INTEGER):
            return IntegerLiteral(value=self._previous.value, location=loc)

        # Float literal
        if self._match(TokenType.FLOAT):
            return FloatLiteral(value=self._previous.value, location=loc)

        # String literal (may be a regular string or an f-string)
        if self._match(TokenType.STRING):
            value = self._previous.value
            # Check if it's an f-string (lexer returns ("fstring", parts) for f-strings)
            if isinstance(value, tuple) and len(value) == 2 and value[0] == "fstring":
                return self._parse_fstring_parts(value[1], loc)
            return StringLiteral(value=value, location=loc)

        # Boolean literals
        if self._match(TokenType.TRUE, TokenType.FALSE):
            return BooleanLiteral(value=self._previous.value, location=loc)

        # None literal
        if self._match(TokenType.NONE):
            return NoneLiteral(location=loc)

        # Mathematical constants
        if self._match(TokenType.PI):
            return FloatLiteral(value=self._previous.value, location=loc)

        if self._match(TokenType.INFINITY):
            return FloatLiteral(value=self._previous.value, location=loc)

        if self._match(TokenType.EMPTY_SET):
            return SetLiteral(elements=(), location=loc)

        # List literal or list comprehension
        if self._match(TokenType.LBRACKET):
            return self._parse_list_literal_or_comprehension(loc)

        # Set or dict literal (both use braces)
        if self._match(TokenType.LBRACE):
            return self._parse_set_or_dict_literal(loc)

        # Some(value) - Optional constructor
        if self._match(TokenType.SOME):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Some'")
            value = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after Some value")
            return SomeExpression(value=value, location=loc)

        # Ok(value) - Result success constructor
        if self._match(TokenType.OK):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Ok'")
            value = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after Ok value")
            return OkExpression(value=value, location=loc)

        # Err(error) - Result error constructor
        if self._match(TokenType.ERR):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Err'")
            value = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after Err value")
            return ErrExpression(value=value, location=loc)

        # Match expression
        if self._match(TokenType.MATCH):
            return self._parse_match_expression(loc)

        # Self expression
        if self._match(TokenType.SELF):
            return SelfExpression(location=loc)

        # Identifier (possibly with :: for enum variant access)
        if self._match(TokenType.IDENTIFIER):
            name = self._previous.value
            # Check for enum variant access: Name::Variant
            if self._match(TokenType.DOUBLE_COLON):
                variant_name = self._expect(
                    TokenType.IDENTIFIER, "Expected variant name after '::'"
                ).value
                return EnumVariantAccess(
                    enum_name=name, variant_name=variant_name, location=loc
                )
            # Check for struct literal: Name { field: value, ... }
            if self._check(TokenType.LBRACE) and self._is_struct_literal_context():
                return self._parse_struct_literal(name, loc)
            return Identifier(name=name, location=loc)

        raise self._error(f"Unexpected token: {self._current.type.name}")

    def _parse_fstring_parts(
        self, parts: tuple, loc: SourceLocation
    ) -> FString:
        """
        Parse f-string parts into an FString AST node.

        The lexer returns parts as a tuple of:
        - ("literal", value) for literal parts
        - ("expr", expr_str) for expressions without format spec
        - ("expr", expr_str, format_spec) for expressions with format spec

        We need to parse the expression strings into actual Expression nodes.
        """
        from mathviz.compiler.lexer import Lexer

        fstring_parts: list[FStringPart] = []

        for part in parts:
            if part[0] == "literal":
                fstring_parts.append(FStringLiteral(value=part[1]))
            elif part[0] == "expr":
                expr_str = part[1]
                format_spec = part[2] if len(part) > 2 else None

                # Parse the expression string
                try:
                    expr_tokens = Lexer(expr_str).tokenize()
                    expr_parser = Parser(expr_tokens, source=expr_str)
                    expr = expr_parser._parse_expression()
                    fstring_parts.append(
                        FStringExpression(expression=expr, format_spec=format_spec)
                    )
                except Exception:
                    raise self._error(f"Invalid expression in f-string: {expr_str}")

        return FString(parts=tuple(fstring_parts), location=loc)

    def _parse_list_literal_or_comprehension(self, loc: SourceLocation) -> Expression:
        """
        Parse a list literal or list comprehension.

        Distinguishes between:
            []           -> empty list
            [1, 2, 3]    -> list literal
            [x^2 for x in 0..10]           -> list comprehension
            [x for x in items if x > 0]    -> list comprehension with filter
        """
        # Empty list
        if self._match(TokenType.RBRACKET):
            return ListLiteral(elements=(), location=loc)

        # Parse first expression
        first = self._parse_expression()

        # Check if it's a comprehension (has 'for' keyword)
        if self._check(TokenType.FOR):
            clauses = self._parse_comprehension_clauses()
            self._expect(TokenType.RBRACKET, "Expected ']' after list comprehension")
            return ListComprehension(element=first, clauses=tuple(clauses), location=loc)

        # Regular list literal
        elements: list[Expression] = [first]
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RBRACKET):
                break  # Trailing comma
            elements.append(self._parse_expression())

        self._expect(TokenType.RBRACKET, "Expected ']' after list elements")
        return ListLiteral(elements=tuple(elements), location=loc)

    def _parse_set_or_dict_literal(self, loc: SourceLocation) -> Expression:
        """
        Parse a set, dictionary, set comprehension, or dict comprehension.

        Distinguishes between:
            {}                           -> empty dict
            {1, 2, 3}                    -> set literal
            {"a": 1, "b": 2}             -> dict literal
            {x % 10 for x in items}      -> set comprehension
            {k: v for (k, v) in items}   -> dict comprehension
        """
        # Empty braces -> empty dict
        if self._match(TokenType.RBRACE):
            return DictLiteral(pairs=(), location=loc)

        first = self._parse_expression()

        # Check if it's a dict or dict comprehension (has colon after first expression)
        if self._match(TokenType.COLON):
            first_value = self._parse_expression()

            # Check for dict comprehension
            if self._check(TokenType.FOR):
                clauses = self._parse_comprehension_clauses()
                self._expect(TokenType.RBRACE, "Expected '}' after dict comprehension")
                return DictComprehension(
                    key=first, value=first_value, clauses=tuple(clauses), location=loc
                )

            # Regular dictionary literal
            pairs: list[tuple[Expression, Expression]] = [(first, first_value)]
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACE):
                    break  # Trailing comma
                key = self._parse_expression()
                self._expect(TokenType.COLON, "Expected ':' in dictionary literal")
                value = self._parse_expression()
                pairs.append((key, value))

            self._expect(TokenType.RBRACE, "Expected '}' after dictionary")
            return DictLiteral(pairs=tuple(pairs), location=loc)

        # Check for set comprehension
        if self._check(TokenType.FOR):
            clauses = self._parse_comprehension_clauses()
            self._expect(TokenType.RBRACE, "Expected '}' after set comprehension")
            return SetComprehension(element=first, clauses=tuple(clauses), location=loc)

        # Regular set literal
        elements: list[Expression] = [first]
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RBRACE):
                break  # Trailing comma
            elements.append(self._parse_expression())

        self._expect(TokenType.RBRACE, "Expected '}' after set elements")
        return SetLiteral(elements=tuple(elements), location=loc)

    def _parse_comprehension_clauses(self) -> list[ComprehensionClause]:
        """
        Parse one or more comprehension clauses (for...in...if).

        Examples:
            for x in items
            for x in items if x > 0
            for x in 0..3 for y in 0..3
        """
        clauses: list[ComprehensionClause] = []

        while self._match(TokenType.FOR):
            clause_loc = self._previous.location

            # Parse variable (could be simple identifier or tuple pattern)
            variable = self._expect(TokenType.IDENTIFIER, "Expected variable name after 'for'").value

            self._expect(TokenType.IN, "Expected 'in' after variable in comprehension")

            iterable = self._parse_expression()

            # Check for optional 'if' condition
            condition: Optional[Expression] = None
            if self._match(TokenType.IF):
                condition = self._parse_expression()

            clauses.append(ComprehensionClause(
                variable=variable,
                iterable=iterable,
                condition=condition,
                location=clause_loc,
            ))

        return clauses

    def _parse_pipe_lambda(self, loc: SourceLocation) -> PipeLambda:
        """
        Parse a pipe-style lambda expression.

        Examples:
            |x| x * 2
            |x, y| x + y
            |acc, x| acc + x

        The opening pipe has already been consumed.
        """
        # Parse parameters
        params: list[str] = []

        # Empty params: || expr
        if self._match(TokenType.PIPE):
            body = self._parse_expression()
            return PipeLambda(parameters=(), body=body, location=loc)

        # Parse parameter list
        while True:
            param_name = self._expect(TokenType.IDENTIFIER, "Expected parameter name in lambda").value
            params.append(param_name)

            if not self._match(TokenType.COMMA):
                break

        # Expect closing pipe
        self._expect(TokenType.PIPE, "Expected '|' after lambda parameters")

        # Parse the body expression
        body = self._parse_expression()

        return PipeLambda(parameters=tuple(params), body=body, location=loc)

    def _is_struct_literal_context(self) -> bool:
        """
        Check if we're in a context where a struct literal is expected.

        This distinguishes between:
            Point { x: 1.0, y: 2.0 }  // struct literal
            if cond { ... }           // block (not struct literal)

        We check if the content looks like "identifier: expression".
        """
        if not self._check(TokenType.LBRACE):
            return False

        # Look ahead: if we see IDENTIFIER COLON after LBRACE, it's a struct literal
        # Save position
        saved_pos = self.pos

        # Skip the LBRACE
        self.pos += 1
        self._skip_newlines()

        result = False
        if self._check(TokenType.IDENTIFIER):
            self.pos += 1
            if self._check(TokenType.COLON):
                result = True

        # Restore position
        self.pos = saved_pos
        return result

    def _parse_struct_literal(self, struct_name: str, loc: SourceLocation) -> StructLiteral:
        """
        Parse a struct literal with named fields.

        Example:
            Point { x: 1.0, y: 2.0 }
        """
        self._expect(TokenType.LBRACE, "Expected '{' for struct literal")
        self._skip_newlines()

        fields: list[tuple[str, Expression]] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            field_name = self._expect(TokenType.IDENTIFIER, "Expected field name").value
            self._expect(TokenType.COLON, "Expected ':' after field name")
            field_value = self._parse_expression()
            fields.append((field_name, field_value))

            self._skip_newlines()
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close struct literal")

        return StructLiteral(
            struct_name=struct_name,
            fields=tuple(fields),
            location=loc,
        )

    def _parse_arguments(self) -> tuple[list[Expression], list[KeywordArgument]]:
        """Parse function call arguments (supports multiline and keyword args).

        Returns a tuple of (positional_args, keyword_args).
        """
        positional_args: list[Expression] = []
        keyword_args: list[KeywordArgument] = []
        seen_keyword = False

        self._skip_newlines()
        if not self._check(TokenType.RPAREN):
            while True:
                self._skip_newlines()

                # Check for keyword argument: identifier followed by colon
                if (self._check(TokenType.IDENTIFIER) and
                    self.pos + 1 < len(self.tokens) and
                    self.tokens[self.pos + 1].type == TokenType.COLON):
                    # It's a keyword argument
                    loc = self._current.location
                    name = self._current.value
                    self._advance()  # consume identifier
                    self._advance()  # consume colon
                    value = self._parse_expression()
                    keyword_args.append(KeywordArgument(name=name, value=value, location=loc))
                    seen_keyword = True
                else:
                    # Positional argument
                    if seen_keyword:
                        self._error("Positional argument cannot follow keyword argument")
                    positional_args.append(self._parse_expression())

                self._skip_newlines()
                if not self._match(TokenType.COMMA):
                    break
                self._skip_newlines()

        self._skip_newlines()
        return positional_args, keyword_args

    def _parse_expression_list(self, end_token: TokenType) -> list[Expression]:
        """Parse a comma-separated list of expressions."""
        elements: list[Expression] = []

        if not self._check(end_token):
            while True:
                elements.append(self._parse_expression())
                if not self._match(TokenType.COMMA):
                    break
                if self._check(end_token):
                    break  # Trailing comma

        return elements

    def _parse_lambda_body(self) -> Expression | Block:
        """Parse the body of a lambda expression."""
        if self._check(TokenType.LBRACE):
            return self._parse_block()
        return self._parse_expression()

    def _expression_to_parameter(self, expr: Expression) -> Parameter:
        """Convert an expression to a parameter (for lambda parsing)."""
        if isinstance(expr, Identifier):
            return Parameter(name=expr.name, location=expr.location)
        raise self._error("Expected parameter name in lambda")

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def _parse_match_expression(self, loc: SourceLocation) -> MatchExpression:
        """
        Parse a match expression.

        Handles:
            match value {
                0 -> println("zero")
                1 -> println("one")
                n where n > 0 -> println("positive: {}", n)
                _ -> println("fallback")
            }
        """
        # Parse the subject expression
        subject = self._parse_expression()

        self._skip_newlines()
        self._expect(TokenType.LBRACE, "Expected '{' after match subject")
        self._skip_newlines()

        # Parse match arms
        arms: list[MatchArm] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            arm = self._parse_match_arm()
            arms.append(arm)
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close match expression")

        return MatchExpression(
            subject=subject,
            arms=tuple(arms),
            location=loc,
        )

    def _parse_match_arm(self) -> MatchArm:
        """
        Parse a single match arm.

        Handles:
            pattern -> body
            pattern where guard -> body
        """
        loc = self._current.location

        # Parse the pattern
        pattern = self._parse_pattern()

        # Parse optional guard
        guard: Optional[Expression] = None
        if self._match(TokenType.WHERE):
            guard = self._parse_expression()

        # Expect arrow separator
        self._expect(TokenType.THIN_ARROW, "Expected '->' after pattern")

        # Parse body (expression or block)
        if self._check(TokenType.LBRACE):
            body: Expression | Block = self._parse_block()
        else:
            body = self._parse_expression()

        return MatchArm(
            pattern=pattern,
            guard=guard,
            body=body,
            location=loc,
        )

    def _parse_pattern(self) -> Pattern:
        """
        Parse a pattern with full support for advanced pattern matching.

        Pattern precedence (lowest to highest):
            1. Or patterns: a | b | c
            2. Binding patterns: x @ pattern
            3. Range patterns: 1..10, 'a'..'z'
            4. Primary patterns: literals, identifiers, tuples, lists, constructors

        Handles:
            - Literal patterns: 0, "hello", true, None
            - Identifier patterns: n, x (binds value)
            - Wildcard pattern: _
            - Tuple patterns: (x, y, z), (0, 0)
            - Constructor patterns: Circle(r), Some(x), None
            - Range patterns: 1..10, 'a'..'z', 0..=100
            - Or patterns: "a" | "b" | "c"
            - Binding patterns: x @ 1..100, list @ [first, ..rest]
            - Rest patterns: ..rest, ..
            - List patterns: [a, b, c], [first, ..rest]
        """
        return self._parse_or_pattern()

    def _parse_or_pattern(self) -> Pattern:
        """Parse or patterns: pattern1 | pattern2 | pattern3"""
        loc = self._current.location
        left = self._parse_binding_pattern()

        if self._check(TokenType.PIPE):
            patterns = [left]
            while self._match(TokenType.PIPE):
                patterns.append(self._parse_binding_pattern())
            return OrPattern(patterns=tuple(patterns), location=loc)

        return left

    def _parse_binding_pattern(self) -> Pattern:
        """Parse binding patterns: name @ pattern"""
        loc = self._current.location

        # Check if this is an identifier followed by @
        if self._check(TokenType.IDENTIFIER):
            # Look ahead to see if there's an @ after the identifier
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.AT:
                name = self._advance().value  # consume identifier
                self._advance()  # consume @
                inner = self._parse_range_pattern()
                return BindingPattern(name=name, pattern=inner, location=loc)

        return self._parse_range_pattern()

    def _parse_range_pattern(self) -> Pattern:
        """Parse range patterns: start..end, start..=end"""
        left = self._parse_primary_pattern()

        # Check for range pattern (..)
        if self._check(TokenType.DOT):
            # Look ahead for second dot
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.DOT:
                self._advance()  # consume first dot
                self._advance()  # consume second dot
                inclusive = self._match(TokenType.ASSIGN)  # ..=
                end = self._parse_primary_pattern()

                # Convert LiteralPattern to Expression for RangePattern
                start_expr = self._pattern_to_expr(left)
                end_expr = self._pattern_to_expr(end)

                if start_expr is None or end_expr is None:
                    raise self._error("Range patterns require literal start and end values")

                return RangePattern(
                    start=start_expr,
                    end=end_expr,
                    inclusive=inclusive,
                    location=left.location,
                )

        return left

    def _pattern_to_expr(self, pattern: Pattern) -> Optional[Expression]:
        """Convert a literal pattern to an expression for range patterns."""
        if isinstance(pattern, LiteralPattern):
            return pattern.value
        return None

    def _parse_primary_pattern(self) -> Pattern:
        """Parse primary patterns (literals, identifiers, tuples, lists, constructors)."""
        loc = self._current.location

        # Rest pattern: .. or ..name
        if self._check(TokenType.DOT):
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.DOT:
                self._advance()  # consume first dot
                self._advance()  # consume second dot
                # Check for optional name
                name: Optional[str] = None
                if self._check(TokenType.IDENTIFIER):
                    name = self._advance().value
                return RestPattern(name=name, location=loc)

        # Wildcard pattern: _
        if self._match(TokenType.UNDERSCORE):
            return IdentifierPattern(name="_", is_wildcard=True, location=loc)

        # Tuple pattern: (...)
        if self._match(TokenType.LPAREN):
            return self._parse_tuple_pattern(loc)

        # List pattern: [...]
        if self._match(TokenType.LBRACKET):
            return self._parse_list_pattern(loc)

        # Literal patterns
        if self._match(TokenType.INTEGER):
            return LiteralPattern(
                value=IntegerLiteral(value=self._previous.value, location=loc),
                location=loc,
            )

        if self._match(TokenType.FLOAT):
            return LiteralPattern(
                value=FloatLiteral(value=self._previous.value, location=loc),
                location=loc,
            )

        if self._match(TokenType.STRING):
            return LiteralPattern(
                value=StringLiteral(value=self._previous.value, location=loc),
                location=loc,
            )

        if self._match(TokenType.TRUE, TokenType.FALSE):
            return LiteralPattern(
                value=BooleanLiteral(value=self._previous.value, location=loc),
                location=loc,
            )

        if self._match(TokenType.NONE):
            return LiteralPattern(
                value=NoneLiteral(location=loc),
                location=loc,
            )

        # Constructor or identifier pattern
        if self._match(TokenType.IDENTIFIER):
            name = self._previous.value

            # Check for enum pattern: Name::Variant or Name::Variant(args)
            if self._match(TokenType.DOUBLE_COLON):
                variant_name = self._expect(
                    TokenType.IDENTIFIER, "Expected variant name after '::'"
                ).value
                bindings: list[Pattern] = []
                if self._match(TokenType.LPAREN):
                    if not self._check(TokenType.RPAREN):
                        bindings = self._parse_pattern_list()
                    self._expect(TokenType.RPAREN, "Expected ')' after variant bindings")
                return EnumPattern(
                    enum_name=name,
                    variant_name=variant_name,
                    bindings=tuple(bindings),
                    location=loc,
                )

            # Check for constructor pattern: Name(args)
            if self._match(TokenType.LPAREN):
                args = self._parse_pattern_list()
                self._expect(TokenType.RPAREN, "Expected ')' after constructor arguments")
                return ConstructorPattern(name=name, args=tuple(args), location=loc)

            # Simple identifier pattern (binds the value)
            return IdentifierPattern(name=name, is_wildcard=False, location=loc)

        # Optional/Result constructors as patterns
        if self._match(TokenType.SOME):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Some'")
            inner = self._parse_pattern()
            self._expect(TokenType.RPAREN, "Expected ')' after Some pattern")
            return ConstructorPattern(name="Some", args=(inner,), location=loc)

        if self._match(TokenType.OK):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Ok'")
            inner = self._parse_pattern()
            self._expect(TokenType.RPAREN, "Expected ')' after Ok pattern")
            return ConstructorPattern(name="Ok", args=(inner,), location=loc)

        if self._match(TokenType.ERR):
            self._expect(TokenType.LPAREN, "Expected '(' after 'Err'")
            inner = self._parse_pattern()
            self._expect(TokenType.RPAREN, "Expected ')' after Err pattern")
            return ConstructorPattern(name="Err", args=(inner,), location=loc)

        raise self._error(f"Expected pattern, got {self._current.type.name}")

    def _parse_tuple_pattern(self, loc: SourceLocation) -> Pattern:
        """
        Parse a tuple pattern: (x, y, z) or (0, 0) or (x, _).

        Handles:
            ()              -> empty tuple pattern
            (pattern)       -> grouped pattern (not a tuple)
            (pattern,)      -> single-element tuple pattern
            (p1, p2, ...)   -> multi-element tuple pattern
            (first, ..)     -> tuple with rest pattern
            (.., last)      -> tuple with leading rest pattern
            (first, .., last) -> tuple with middle rest pattern
        """
        # Empty tuple
        if self._check(TokenType.RPAREN):
            self._advance()
            return TuplePattern(elements=(), location=loc)

        # Check for leading rest pattern: (.., ...)
        if self._check(TokenType.DOT):
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.DOT:
                first = self._parse_pattern()  # This will parse ..rest or ..
                if self._match(TokenType.COMMA):
                    # More elements after rest
                    elements = [first]
                    while True:
                        elements.append(self._parse_pattern())
                        if not self._match(TokenType.COMMA):
                            break
                        if self._check(TokenType.RPAREN):
                            break
                    self._expect(TokenType.RPAREN, "Expected ')' after tuple pattern")
                    return TuplePattern(elements=tuple(elements), location=loc)
                else:
                    # Just the rest pattern in parens
                    self._expect(TokenType.RPAREN, "Expected ')' after pattern")
                    return TuplePattern(elements=(first,), location=loc)

        # Parse first pattern
        first = self._parse_pattern()

        # Check for comma -> tuple pattern
        if self._match(TokenType.COMMA):
            # Check if it's a trailing comma for single-element tuple
            if self._check(TokenType.RPAREN):
                self._advance()
                return TuplePattern(elements=(first,), location=loc)

            # Multiple elements
            elements = [first]
            while True:
                elements.append(self._parse_pattern())
                if not self._match(TokenType.COMMA):
                    break
                # Check for trailing comma
                if self._check(TokenType.RPAREN):
                    break

            self._expect(TokenType.RPAREN, "Expected ')' after tuple pattern")
            return TuplePattern(elements=tuple(elements), location=loc)

        # Single pattern in parens - just return it (grouping)
        self._expect(TokenType.RPAREN, "Expected ')' after pattern")
        return first

    def _parse_list_pattern(self, loc: SourceLocation) -> Pattern:
        """
        Parse a list pattern: [x, y, z] or [first, ..rest] or [.., last].

        Handles:
            []                  -> empty list pattern
            [x]                 -> single element
            [x, y, z]           -> exact elements
            [first, ..rest]     -> first element and rest
            [first, second, ..] -> at least two elements
            [.., last]          -> last element only
            [first, .., last]   -> first and last elements
        """
        # Empty list
        if self._check(TokenType.RBRACKET):
            self._advance()
            return ListPattern(elements=(), location=loc)

        elements: list[Pattern] = []

        while True:
            elements.append(self._parse_pattern())
            if not self._match(TokenType.COMMA):
                break
            # Check for trailing comma
            if self._check(TokenType.RBRACKET):
                break

        self._expect(TokenType.RBRACKET, "Expected ']' after list pattern")
        return ListPattern(elements=tuple(elements), location=loc)

    def _parse_pattern_list(self) -> list[Pattern]:
        """Parse a comma-separated list of patterns."""
        patterns: list[Pattern] = []

        if not self._check(TokenType.RPAREN):
            while True:
                patterns.append(self._parse_pattern())
                if not self._match(TokenType.COMMA):
                    break
                # Allow trailing comma
                if self._check(TokenType.RPAREN):
                    break

        return patterns

    # -------------------------------------------------------------------------
    # OOP Constructs (Structs, Traits, Enums, Impl Blocks)
    # -------------------------------------------------------------------------

    def _parse_struct_def(self) -> StructDef:
        """
        Parse a struct definition with optional generics.

        Handles:
            struct Point {
                x: Float
                y: Float
            }

            struct Box<T> {
                value: T
            }

            struct Pair<A, B> {
                first: A
                second: B
            }
        """
        loc = self._current.location
        self._advance()  # consume 'struct'

        name = self._expect(TokenType.IDENTIFIER, "Expected struct name").value

        # Optional type parameters
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        self._skip_newlines()
        self._expect(TokenType.LBRACE, "Expected '{' after struct name")
        self._skip_newlines()

        fields: list[StructField] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            field = self._parse_struct_field()
            fields.append(field)
            self._skip_newlines()
            # Allow optional comma or newline between fields
            self._match(TokenType.COMMA)
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close struct definition")

        return StructDef(
            name=name,
            fields=tuple(fields),
            type_params=type_params,
            where_clause=where_clause,
            location=loc,
        )

    def _parse_struct_field(self) -> StructField:
        """Parse a single struct field."""
        loc = self._current.location

        # Check for visibility modifier
        visibility = Visibility.PRIVATE
        if self._match(TokenType.PUB):
            visibility = Visibility.PUBLIC

        name = self._expect(TokenType.IDENTIFIER, "Expected field name").value
        self._expect(TokenType.COLON, "Expected ':' after field name")
        type_annotation = self._parse_type_annotation()

        return StructField(
            name=name,
            type_annotation=type_annotation,
            visibility=visibility,
            location=loc,
        )

    def _parse_impl_block(self) -> ImplBlock:
        """
        Parse an impl block with optional generics, associated types, and operator traits.

        Handles:
            impl Point {
                fn distance(self, other: Point) -> Float { ... }
                fn origin() -> Point { ... }
            }

            impl Shape for Circle {
                fn area(self) -> Float { ... }
            }

            impl<T> Box<T> {
                fn new(value: T) -> Box<T> { ... }
                fn unwrap(self) -> T { ... }
            }

            impl<T: Display> Box<T> {
                fn print(self) { ... }
            }

            // Operator trait implementations
            impl Add for Vector {
                type Output = Vector
                fn add(self, other: Vector) -> Vector { ... }
            }

            impl Mul<Float> for Vector {
                type Output = Vector
                fn mul(self, scalar: Float) -> Vector { ... }
            }
        """
        loc = self._current.location
        self._advance()  # consume 'impl'

        # Optional type parameters on impl: impl<T> ...
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Parse target type name (or trait name if this is a trait impl)
        first_type = self._expect(TokenType.IDENTIFIER, "Expected type name after 'impl'").value

        # Parse optional type arguments: Box<T> or Mul<Float>
        first_type_args: list[TypeAnnotation] = []
        if self._check(TokenType.LT):
            self._advance()  # consume '<'
            while True:
                type_arg = self._parse_type_annotation()
                first_type_args.append(type_arg)
                if not self._match(TokenType.COMMA):
                    break
            self._expect(TokenType.GT, "Expected '>' to close type arguments")

        # Check for 'for' keyword (trait impl)
        trait_name: Optional[str] = None
        trait_type_args: tuple[TypeAnnotation, ...] = ()
        target_type: str
        target_type_args: list[str] = []

        if self._check(TokenType.FOR):
            self._advance()  # consume 'for'
            # first_type is actually the trait name
            trait_name = first_type
            trait_type_args = tuple(first_type_args)

            target_type = self._expect(TokenType.IDENTIFIER, "Expected type name after 'for'").value
            # Parse type args for the actual target type if present
            if self._check(TokenType.LT):
                self._advance()  # consume '<'
                while True:
                    type_arg = self._expect(TokenType.IDENTIFIER, "Expected type argument").value
                    target_type_args.append(type_arg)
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.GT, "Expected '>' to close type arguments")
        else:
            # Inherent impl (no trait)
            target_type = first_type
            # Convert TypeAnnotation list to string list for target type args
            for type_arg in first_type_args:
                if isinstance(type_arg, SimpleType):
                    target_type_args.append(type_arg.name)
                else:
                    # For now, only support simple type args for inherent impls
                    raise self._error("Complex type arguments not supported for inherent impls")

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        self._skip_newlines()
        self._expect(TokenType.LBRACE, "Expected '{' after impl declaration")
        self._skip_newlines()

        methods: list[Method] = []
        associated_types: list[AssociatedType] = []

        while not self._check(TokenType.RBRACE, TokenType.EOF):
            # Check for associated type declaration: type Output = Vector
            if self._check(TokenType.IDENTIFIER) and self._current.value == "type":
                assoc_type = self._parse_associated_type()
                associated_types.append(assoc_type)
            else:
                method = self._parse_method()
                methods.append(method)
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close impl block")

        return ImplBlock(
            target_type=target_type,
            trait_name=trait_name,
            trait_type_args=trait_type_args,
            methods=tuple(methods),
            associated_types=tuple(associated_types),
            type_params=type_params,
            target_type_args=tuple(target_type_args),
            where_clause=where_clause,
            location=loc,
        )

    def _parse_associated_type(self) -> AssociatedType:
        """
        Parse an associated type declaration in an impl block.

        Syntax:
            type Output = Vector
            type Item = Int
        """
        loc = self._current.location
        self._advance()  # consume 'type' (which is an identifier, not a keyword)

        name = self._expect(TokenType.IDENTIFIER, "Expected associated type name").value
        self._expect(TokenType.ASSIGN, "Expected '=' after associated type name")
        type_value = self._parse_type_annotation()

        # Skip optional newline
        self._match(TokenType.NEWLINE)

        return AssociatedType(
            name=name,
            type_value=type_value,
            location=loc,
        )

    def _parse_method(self) -> Method:
        """
        Parse a method definition within an impl block.

        Handles:
            fn distance(self, other: Point) -> Float { ... }
            pub fn greet(self) { ... }
            fn origin() -> Point { ... }  # Static method
        """
        loc = self._current.location

        # Check for visibility modifier
        visibility = Visibility.PRIVATE
        if self._match(TokenType.PUB):
            visibility = Visibility.PUBLIC

        self._expect(TokenType.FN, "Expected 'fn' for method definition")
        name = self._expect(TokenType.IDENTIFIER, "Expected method name").value

        # Optional type parameters for generic methods
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Parameters
        self._expect(TokenType.LPAREN, "Expected '(' after method name")
        params, has_self = self._parse_method_parameters()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")

        # Optional return type
        return_type: Optional[TypeAnnotation] = None
        if self._match(TokenType.THIN_ARROW):
            return_type = self._parse_type_annotation()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        # Body
        self._skip_newlines()
        body = self._parse_block()

        return Method(
            name=name,
            parameters=tuple(params),
            return_type=return_type,
            body=body,
            type_params=type_params,
            where_clause=where_clause,
            visibility=visibility,
            has_self=has_self,
            location=loc,
        )

    def _parse_method_parameters(self) -> tuple[list[Parameter], bool]:
        """
        Parse method parameters, handling 'self' as first parameter.

        Returns:
            A tuple of (parameters, has_self).
        """
        params: list[Parameter] = []
        has_self = False

        if self._check(TokenType.RPAREN):
            return params, has_self

        # Check if first parameter is 'self'
        if self._check(TokenType.SELF):
            loc = self._current.location
            self._advance()  # consume 'self'
            has_self = True
            # 'self' is implicit, we don't add it to params for code gen
            # but we track that it exists

            if not self._match(TokenType.COMMA):
                return params, has_self

        # Parse remaining parameters (reuse existing logic)
        while True:
            loc = self._current.location
            name = self._expect(TokenType.IDENTIFIER, "Expected parameter name").value

            type_annotation: Optional[TypeAnnotation] = None
            if self._match(TokenType.COLON):
                type_annotation = self._parse_type_annotation()

            default_value: Optional[Expression] = None
            if self._match(TokenType.ASSIGN):
                default_value = self._parse_expression()

            params.append(
                Parameter(
                    name=name,
                    type_annotation=type_annotation,
                    default_value=default_value,
                    location=loc,
                )
            )

            if not self._match(TokenType.COMMA):
                break

        return params, has_self

    def _parse_trait_def(self) -> TraitDef:
        """
        Parse a trait definition with optional generics.

        Handles:
            trait Shape {
                fn area(self) -> Float
                fn perimeter(self) -> Float
            }

            trait Container<T> {
                fn get(self) -> T
                fn set(self, value: T)
            }

            trait Comparable<T> {
                fn compare(self, other: T) -> Int
            }
        """
        loc = self._current.location
        self._advance()  # consume 'trait'

        name = self._expect(TokenType.IDENTIFIER, "Expected trait name").value

        # Optional type parameters
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        self._skip_newlines()
        self._expect(TokenType.LBRACE, "Expected '{' after trait name")
        self._skip_newlines()

        methods: list[TraitMethod] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            method = self._parse_trait_method()
            methods.append(method)
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close trait definition")

        return TraitDef(
            name=name,
            methods=tuple(methods),
            type_params=type_params,
            where_clause=where_clause,
            location=loc,
        )

    def _parse_trait_method(self) -> TraitMethod:
        """
        Parse a trait method signature (possibly with default implementation).

        Handles:
            fn area(self) -> Float
            fn perimeter(self) -> Float { ... }  # With default impl
            fn transform<U>(self, f: (T) -> U) -> U  # Generic method
        """
        loc = self._current.location

        self._expect(TokenType.FN, "Expected 'fn' for trait method")
        name = self._expect(TokenType.IDENTIFIER, "Expected method name").value

        # Optional type parameters for generic methods
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Parameters
        self._expect(TokenType.LPAREN, "Expected '(' after method name")
        params, has_self = self._parse_method_parameters()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")

        # Optional return type
        return_type: Optional[TypeAnnotation] = None
        if self._match(TokenType.THIN_ARROW):
            return_type = self._parse_type_annotation()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        # Check for default implementation
        self._skip_newlines()
        has_default_impl = False
        default_body: Optional[Block] = None
        if self._check(TokenType.LBRACE):
            has_default_impl = True
            default_body = self._parse_block()

        return TraitMethod(
            name=name,
            parameters=tuple(params),
            return_type=return_type,
            type_params=type_params,
            where_clause=where_clause,
            has_self=has_self,
            has_default_impl=has_default_impl,
            default_body=default_body,
            location=loc,
        )

    def _parse_enum_def(self) -> EnumDef:
        """
        Parse an enum definition with optional generics.

        Handles:
            enum Shape {
                Circle(Float)
                Rectangle(Float, Float)
                Point
            }

            enum Option<T> {
                Some(T)
                None
            }

            enum Result<T, E> {
                Ok(T)
                Err(E)
            }
        """
        loc = self._current.location
        self._advance()  # consume 'enum'

        name = self._expect(TokenType.IDENTIFIER, "Expected enum name").value

        # Optional type parameters
        type_params: tuple[TypeParameter, ...] = ()
        if self._check(TokenType.LT):
            type_params = self._parse_type_parameters()

        # Optional where clause
        where_clause: Optional[WhereClause] = None
        if self._check(TokenType.WHERE):
            where_clause = self._parse_where_clause()

        self._skip_newlines()
        self._expect(TokenType.LBRACE, "Expected '{' after enum name")
        self._skip_newlines()

        variants: list[EnumVariant] = []
        while not self._check(TokenType.RBRACE, TokenType.EOF):
            variant = self._parse_enum_variant()
            variants.append(variant)
            self._skip_newlines()
            # Allow optional comma between variants
            self._match(TokenType.COMMA)
            self._skip_newlines()

        self._expect(TokenType.RBRACE, "Expected '}' to close enum definition")

        return EnumDef(
            name=name,
            variants=tuple(variants),
            type_params=type_params,
            where_clause=where_clause,
            location=loc,
        )

    def _parse_enum_variant(self) -> EnumVariant:
        """
        Parse a single enum variant.

        Handles:
            Circle(Float)
            Rectangle(Float, Float)
            Point
        """
        loc = self._current.location

        name = self._expect(TokenType.IDENTIFIER, "Expected variant name").value

        # Check for associated data
        fields: list[TypeAnnotation] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                while True:
                    fields.append(self._parse_type_annotation())
                    if not self._match(TokenType.COMMA):
                        break
            self._expect(TokenType.RPAREN, "Expected ')' after variant fields")

        return EnumVariant(name=name, fields=tuple(fields), location=loc)

    def _parse_enum_pattern(self, enum_name: str, loc: SourceLocation) -> EnumPattern:
        """
        Parse an enum variant pattern.

        Example:
            Shape::Circle(r)
            Shape::Rectangle(w, h)
            Shape::Point
        """
        # We already have enum_name, now parse ::Variant
        self._expect(TokenType.DOUBLE_COLON, "Expected '::' for enum variant pattern")
        variant_name = self._expect(TokenType.IDENTIFIER, "Expected variant name").value

        # Check for bindings
        bindings: list[Pattern] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                bindings = self._parse_pattern_list()
            self._expect(TokenType.RPAREN, "Expected ')' after variant bindings")

        return EnumPattern(
            enum_name=enum_name,
            variant_name=variant_name,
            bindings=tuple(bindings),
            location=loc,
        )


def parse(tokens: list[Token]) -> Program:
    """
    Convenience function to parse tokens into an AST.

    Args:
        tokens: List of tokens from the lexer

    Returns:
        The root Program AST node
    """
    return Parser(tokens).parse()
