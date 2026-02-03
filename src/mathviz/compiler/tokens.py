"""
Token definitions for the MathViz lexer.

This module defines all token types recognized by the MathViz language,
including keywords, operators (both standard and mathematical), and literals.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from mathviz.utils.errors import SourceLocation


class TokenType(Enum):
    """Enumeration of all token types in MathViz."""

    # End of file
    EOF = auto()

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()

    # F-string tokens
    FSTRING_START = auto()      # f" or f'
    FSTRING_PART = auto()       # literal parts between {}
    FSTRING_EXPR_START = auto() # {
    FSTRING_EXPR_END = auto()   # }
    FSTRING_END = auto()        # closing quote

    # Identifiers
    IDENTIFIER = auto()

    # Keywords
    LET = auto()
    MUT = auto()  # mut keyword for mutable variables
    CONST = auto()  # const keyword for compile-time constants
    FN = auto()
    CLASS = auto()
    SCENE = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    LOOP = auto()  # Infinite loop
    RETURN = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    IN = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()

    # Pattern matching
    MATCH = auto()      # match keyword
    WHERE = auto()      # where for guards
    UNDERSCORE = auto() # _ wildcard

    # I/O and formatting
    PRINT = auto()         # print()
    PRINTLN = auto()       # println() with newline
    # Note: read_file, write_file are NOT keywords - they are regular functions

    # Manim (SCENE already defined above in keywords)
    ANIMATE = auto()       # animation block
    PLAY = auto()          # play animation
    WAIT = auto()          # wait/pause

    # Module system
    USE = auto()           # use/import module
    MOD = auto()           # module definition
    PUB = auto()           # public visibility

    # OOP constructs
    STRUCT = auto()        # struct keyword for lightweight data types
    IMPL = auto()          # impl keyword for implementation blocks
    TRAIT = auto()         # trait keyword for interfaces
    ENUM = auto()          # enum keyword for algebraic data types
    SELF = auto()          # self keyword for method receiver

    # Decorators
    JIT = auto()            # @jit decorator for Numba optimization
    NJIT = auto()           # @njit decorator (nopython=True by default)
    VECTORIZE = auto()      # @vectorize for ufuncs
    PARALLEL = auto()       # parallel execution hint

    # Async/await
    ASYNC = auto()          # async keyword for async functions
    AWAIT = auto()          # await keyword for awaiting futures

    # Type keywords
    TYPE_INT = auto()
    TYPE_FLOAT = auto()
    TYPE_BOOL = auto()
    TYPE_STRING = auto()
    TYPE_LIST = auto()
    TYPE_SET = auto()
    TYPE_DICT = auto()
    TYPE_TUPLE = auto()
    TYPE_OPTIONAL = auto()
    TYPE_RESULT = auto()

    # Optional and Result constructors
    SOME = auto()           # Some(value)
    OK = auto()             # Ok(value)
    ERR = auto()            # Err(error)

    # Question mark operator (for ? unwrap/propagate)
    QUESTION = auto()       # ?

    # Arithmetic operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    DOUBLE_SLASH = auto()  # //
    PERCENT = auto()       # %
    CARET = auto()         # ^ (exponentiation in MathViz)
    DOUBLE_STAR = auto()   # ** (also exponentiation)

    # Comparison operators
    EQ = auto()            # ==
    NE = auto()            # !=
    LT = auto()            # <
    GT = auto()            # >
    LE = auto()            # <=
    GE = auto()            # >=

    # Assignment
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=

    # Mathematical set operators (Unicode)
    ELEMENT_OF = auto()       # ∈
    NOT_ELEMENT_OF = auto()   # ∉
    SUBSET = auto()           # ⊆
    SUPERSET = auto()         # ⊇
    PROPER_SUBSET = auto()    # ⊂
    PROPER_SUPERSET = auto()  # ⊃
    UNION = auto()            # ∪
    INTERSECTION = auto()     # ∩
    SET_DIFF = auto()         # ∖ or \
    EMPTY_SET = auto()        # ∅

    # Other mathematical symbols
    INFINITY = auto()         # ∞
    APPROX = auto()           # ≈
    NOT_EQUAL_MATH = auto()   # ≠
    LE_MATH = auto()          # ≤
    GE_MATH = auto()          # ≥
    FORALL = auto()           # ∀
    EXISTS = auto()           # ∃
    ARROW = auto()            # →
    DOUBLE_ARROW = auto()     # ⇒
    MAPS_TO = auto()          # ↦
    PI = auto()               # π
    SQRT = auto()             # √
    SUM = auto()              # ∑
    PRODUCT = auto()          # ∏
    INTEGRAL = auto()         # ∫

    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]

    # Punctuation
    COMMA = auto()         # ,
    DOT = auto()           # .
    COLON = auto()         # :
    DOUBLE_COLON = auto()  # :: (enum variant access)
    SEMICOLON = auto()     # ;
    THIN_ARROW = auto()    # ->
    FAT_ARROW = auto()     # =>
    AT = auto()            # @ (decorator)
    PIPE = auto()          # | (for pipe lambdas)

    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    COMMENT = auto()
    DOC_COMMENT = auto()  # /// or /** */

    # Test framework attributes
    TEST = auto()          # @test
    SHOULD_PANIC = auto()  # @should_panic


# Mapping of keywords to token types
KEYWORDS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "mut": TokenType.MUT,
    "const": TokenType.CONST,
    "fn": TokenType.FN,
    "class": TokenType.CLASS,
    "scene": TokenType.SCENE,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "elif": TokenType.ELIF,
    "for": TokenType.FOR,
    "while": TokenType.WHILE,
    "loop": TokenType.LOOP,
    "return": TokenType.RETURN,
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    "in": TokenType.IN,
    "true": TokenType.TRUE,
    "True": TokenType.TRUE,
    "false": TokenType.FALSE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "pass": TokenType.PASS,
    # Pattern matching
    "match": TokenType.MATCH,
    "where": TokenType.WHERE,
    "_": TokenType.UNDERSCORE,
    # I/O functions (print/println are keywords, read_file/write_file are regular functions)
    "print": TokenType.PRINT,
    "println": TokenType.PRINTLN,
    # Manim keywords (scene already defined above)
    "animate": TokenType.ANIMATE,
    "play": TokenType.PLAY,
    "wait": TokenType.WAIT,
    # Module system
    "use": TokenType.USE,
    "mod": TokenType.MOD,
    "pub": TokenType.PUB,
    # OOP constructs
    "struct": TokenType.STRUCT,
    "impl": TokenType.IMPL,
    "trait": TokenType.TRAIT,
    "enum": TokenType.ENUM,
    "self": TokenType.SELF,
    # JIT/Numba decorators
    "jit": TokenType.JIT,
    "njit": TokenType.NJIT,
    "vectorize": TokenType.VECTORIZE,
    "parallel": TokenType.PARALLEL,
    # Async/await
    "async": TokenType.ASYNC,
    "await": TokenType.AWAIT,
    # Type keywords
    "Int": TokenType.TYPE_INT,
    "Float": TokenType.TYPE_FLOAT,
    "Bool": TokenType.TYPE_BOOL,
    "String": TokenType.TYPE_STRING,
    "List": TokenType.TYPE_LIST,
    "Set": TokenType.TYPE_SET,
    "Dict": TokenType.TYPE_DICT,
    "Tuple": TokenType.TYPE_TUPLE,
    "Optional": TokenType.TYPE_OPTIONAL,
    "Result": TokenType.TYPE_RESULT,
    "Some": TokenType.SOME,
    "Ok": TokenType.OK,
    "Err": TokenType.ERR,
    # Note: "test" and "should_panic" are NOT keywords
    # They are handled as identifiers in attribute parsing (@test, @should_panic)
    # LaTeX-style math operators (without backslash)
    "cup": TokenType.UNION,              # ∪ union
    "cap": TokenType.INTERSECTION,       # ∩ intersection
    "notin": TokenType.NOT_ELEMENT_OF,   # ∉ not element of
    "subseteq": TokenType.SUBSET,        # ⊆ subset or equal
    "supseteq": TokenType.SUPERSET,      # ⊇ superset or equal
    "subset": TokenType.PROPER_SUBSET,   # ⊂ proper subset
    "supset": TokenType.PROPER_SUPERSET, # ⊃ proper superset
    "setminus": TokenType.SET_DIFF,      # ∖ set difference
    "emptyset": TokenType.EMPTY_SET,     # ∅ empty set
    "infty": TokenType.INFINITY,         # ∞ infinity
    "neq": TokenType.NOT_EQUAL_MATH,     # ≠ not equal
    "leq": TokenType.LE_MATH,            # ≤ less or equal
    "geq": TokenType.GE_MATH,            # ≥ greater or equal
    "forall": TokenType.FORALL,          # ∀ for all
    "exists": TokenType.EXISTS,          # ∃ exists
    "pi": TokenType.PI,                  # π pi
    # Note: sqrt, sin, cos, etc. are NOT keywords - they are functions converted to np.sqrt(), etc.
    # This allows them to be used naturally: sqrt(x) -> np.sqrt(x)
    # Note: 'in' is already a keyword for iteration AND element membership (∈)
}

# Mapping of mathematical Unicode symbols to token types
MATH_SYMBOLS: dict[str, TokenType] = {
    "∈": TokenType.ELEMENT_OF,
    "∉": TokenType.NOT_ELEMENT_OF,
    "⊆": TokenType.SUBSET,
    "⊇": TokenType.SUPERSET,
    "⊂": TokenType.PROPER_SUBSET,
    "⊃": TokenType.PROPER_SUPERSET,
    "∪": TokenType.UNION,
    "∩": TokenType.INTERSECTION,
    "∖": TokenType.SET_DIFF,
    "∅": TokenType.EMPTY_SET,
    "∞": TokenType.INFINITY,
    "≈": TokenType.APPROX,
    "≠": TokenType.NOT_EQUAL_MATH,
    "≤": TokenType.LE_MATH,
    "≥": TokenType.GE_MATH,
    "∀": TokenType.FORALL,
    "∃": TokenType.EXISTS,
    "→": TokenType.ARROW,
    "⇒": TokenType.DOUBLE_ARROW,
    "↦": TokenType.MAPS_TO,
    "π": TokenType.PI,
    "√": TokenType.SQRT,
    "∑": TokenType.SUM,
    "∏": TokenType.PRODUCT,
    "∫": TokenType.INTEGRAL,
}

# Single character operators
SINGLE_CHAR_TOKENS: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "^": TokenType.CARET,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "=": TokenType.ASSIGN,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
    "@": TokenType.AT,
    "\\": TokenType.SET_DIFF,  # ASCII alternative for set difference
    "?": TokenType.QUESTION,
    "|": TokenType.PIPE,  # Pipe for pipe lambdas |x| x * 2
}

# Two character operators (order matters - check these before single char)
# Note: // is NOT here because it's used for comments, not integer division
DOUBLE_CHAR_TOKENS: dict[str, TokenType] = {
    "==": TokenType.EQ,
    "!=": TokenType.NE,
    "<=": TokenType.LE,
    ">=": TokenType.GE,
    "**": TokenType.DOUBLE_STAR,
    "->": TokenType.THIN_ARROW,
    "=>": TokenType.FAT_ARROW,
    "+=": TokenType.PLUS_ASSIGN,
    "-=": TokenType.MINUS_ASSIGN,
    "*=": TokenType.STAR_ASSIGN,
    "/=": TokenType.SLASH_ASSIGN,
    "::": TokenType.DOUBLE_COLON,  # Enum variant access
}


@dataclass(slots=True)
class Token:
    """
    Represents a single token from the source code.

    Attributes:
        type: The type of this token
        value: The literal value (for literals) or lexeme text
        location: Source location of this token
    """

    type: TokenType
    value: Any
    location: SourceLocation

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type.name}, {self.value!r}, {self.location})"
        return f"Token({self.type.name}, {self.location})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        if isinstance(other, TokenType):
            return self.type == other
        return NotImplemented

    @property
    def is_literal(self) -> bool:
        """Check if this token represents a literal value."""
        return self.type in {
            TokenType.INTEGER,
            TokenType.FLOAT,
            TokenType.STRING,
            TokenType.BOOLEAN,
        }

    @property
    def is_operator(self) -> bool:
        """Check if this token represents an operator."""
        return self.type in {
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.STAR,
            TokenType.SLASH,
            TokenType.DOUBLE_SLASH,
            TokenType.PERCENT,
            TokenType.CARET,
            TokenType.DOUBLE_STAR,
            TokenType.EQ,
            TokenType.NE,
            TokenType.LT,
            TokenType.GT,
            TokenType.LE,
            TokenType.GE,
            TokenType.ELEMENT_OF,
            TokenType.NOT_ELEMENT_OF,
            TokenType.SUBSET,
            TokenType.SUPERSET,
            TokenType.UNION,
            TokenType.INTERSECTION,
            TokenType.SET_DIFF,
            TokenType.AND,
            TokenType.OR,
            TokenType.NOT,
        }

    @property
    def is_math_operator(self) -> bool:
        """Check if this token represents a mathematical set operator."""
        return self.type in {
            TokenType.ELEMENT_OF,
            TokenType.NOT_ELEMENT_OF,
            TokenType.SUBSET,
            TokenType.SUPERSET,
            TokenType.PROPER_SUBSET,
            TokenType.PROPER_SUPERSET,
            TokenType.UNION,
            TokenType.INTERSECTION,
            TokenType.SET_DIFF,
        }
