"""
Completion item providers for MathViz LSP.

This module provides various completion providers for different contexts:
- Keywords and language constructs
- Built-in types
- Standard library functions
- Code snippets
- Local symbols
"""

from dataclasses import dataclass

from lsprotocol import types

from mathviz.compiler.tokens import KEYWORDS
from mathviz.lsp.symbols import Symbol, SymbolKind

# MathViz keywords organized by category
KEYWORD_CATEGORIES = {
    "control_flow": ["if", "else", "elif", "for", "while", "match", "break", "continue", "return"],
    "declarations": ["let", "const", "fn", "struct", "enum", "trait", "impl", "class", "scene"],
    "modifiers": ["pub", "async", "await"],
    "types": ["Int", "Float", "Bool", "String", "List", "Set", "Dict", "Tuple", "Optional", "Result"],
    "values": ["true", "false", "None", "Some", "Ok", "Err"],
    "operators": ["and", "or", "not", "in"],
    "imports": ["import", "from", "use", "as", "mod"],
    "other": ["pass", "self", "print", "println"],
}

# Built-in types with their descriptions
BUILTIN_TYPES = {
    "Int": "Integer type (64-bit signed)",
    "Float": "Floating-point type (64-bit IEEE 754)",
    "Bool": "Boolean type (true or false)",
    "String": "UTF-8 string type",
    "List": "Dynamic list/array type",
    "Set": "Unordered set type",
    "Dict": "Key-value dictionary type",
    "Tuple": "Fixed-size tuple type",
    "Optional": "Optional type (Some(value) or None)",
    "Result": "Result type (Ok(value) or Err(error))",
    "Vec": "Mathematical vector type",
    "Mat": "Mathematical matrix type",
    "Array": "N-dimensional array type",
}

# Standard library math functions
STDLIB_MATH = {
    "sqrt": ("sqrt(x: Float) -> Float", "Square root"),
    "sin": ("sin(x: Float) -> Float", "Sine function"),
    "cos": ("cos(x: Float) -> Float", "Cosine function"),
    "tan": ("tan(x: Float) -> Float", "Tangent function"),
    "asin": ("asin(x: Float) -> Float", "Arc sine function"),
    "acos": ("acos(x: Float) -> Float", "Arc cosine function"),
    "atan": ("atan(x: Float) -> Float", "Arc tangent function"),
    "atan2": ("atan2(y: Float, x: Float) -> Float", "Two-argument arc tangent"),
    "exp": ("exp(x: Float) -> Float", "Exponential function (e^x)"),
    "log": ("log(x: Float) -> Float", "Natural logarithm"),
    "log10": ("log10(x: Float) -> Float", "Base-10 logarithm"),
    "log2": ("log2(x: Float) -> Float", "Base-2 logarithm"),
    "pow": ("pow(base: Float, exp: Float) -> Float", "Power function"),
    "abs": ("abs(x: Float) -> Float", "Absolute value"),
    "floor": ("floor(x: Float) -> Int", "Floor function"),
    "ceil": ("ceil(x: Float) -> Int", "Ceiling function"),
    "round": ("round(x: Float) -> Int", "Round to nearest integer"),
    "min": ("min(a: T, b: T) -> T", "Minimum of two values"),
    "max": ("max(a: T, b: T) -> T", "Maximum of two values"),
    "clamp": ("clamp(x: T, min: T, max: T) -> T", "Clamp value to range"),
}

# Standard library string functions
STDLIB_STRING = {
    "len": ("len(s: String) -> Int", "String length"),
    "upper": ("upper(s: String) -> String", "Convert to uppercase"),
    "lower": ("lower(s: String) -> String", "Convert to lowercase"),
    "trim": ("trim(s: String) -> String", "Remove leading/trailing whitespace"),
    "split": ("split(s: String, sep: String) -> List[String]", "Split string by separator"),
    "join": ("join(parts: List[String], sep: String) -> String", "Join strings with separator"),
    "contains": ("contains(s: String, sub: String) -> Bool", "Check if string contains substring"),
    "starts_with": ("starts_with(s: String, prefix: String) -> Bool", "Check string prefix"),
    "ends_with": ("ends_with(s: String, suffix: String) -> Bool", "Check string suffix"),
    "replace": ("replace(s: String, old: String, new: String) -> String", "Replace occurrences"),
}

# Standard library collection functions
STDLIB_COLLECTIONS = {
    "len": ("len(collection) -> Int", "Collection length"),
    "push": ("push(list: List[T], item: T)", "Append item to list"),
    "pop": ("pop(list: List[T]) -> Optional[T]", "Remove and return last item"),
    "insert": ("insert(list: List[T], index: Int, item: T)", "Insert item at index"),
    "remove": ("remove(list: List[T], index: Int) -> T", "Remove item at index"),
    "sort": ("sort(list: List[T]) -> List[T]", "Sort list in ascending order"),
    "reverse": ("reverse(list: List[T]) -> List[T]", "Reverse list"),
    "map": ("map(f: (T) -> U, list: List[T]) -> List[U]", "Apply function to each element"),
    "filter": ("filter(f: (T) -> Bool, list: List[T]) -> List[T]", "Filter elements by predicate"),
    "reduce": ("reduce(f: (T, T) -> T, list: List[T], init: T) -> T", "Reduce list to single value"),
    "zip": ("zip(a: List[T], b: List[U]) -> List[(T, U)]", "Zip two lists together"),
    "enumerate": ("enumerate(list: List[T]) -> List[(Int, T)]", "Add indices to elements"),
    "range": ("range(start: Int, end: Int) -> List[Int]", "Generate integer range"),
}


@dataclass
class CompletionContext:
    """Context information for completion requests."""

    line: int
    character: int
    trigger_character: str | None
    prefix: str  # Text before cursor on the current line
    is_after_dot: bool
    is_in_type_position: bool
    is_in_import: bool
    is_at_line_start: bool


class CompletionProvider:
    """
    Provides completion items for MathViz LSP.

    This class generates context-aware completion suggestions including
    keywords, types, functions, snippets, and symbols.
    """

    def __init__(self) -> None:
        """Initialize the completion provider."""
        self._keyword_completions: list[types.CompletionItem] | None = None
        self._type_completions: list[types.CompletionItem] | None = None
        self._snippet_completions: list[types.CompletionItem] | None = None
        self._stdlib_completions: list[types.CompletionItem] | None = None

    def get_keyword_completions(self) -> list[types.CompletionItem]:
        """
        Get completion items for all MathViz keywords.

        Returns cached items after first call for performance.
        """
        if self._keyword_completions is not None:
            return self._keyword_completions

        completions: list[types.CompletionItem] = []

        # Add keywords with appropriate snippets
        keyword_snippets = {
            "fn": "fn ${1:name}(${2:params}) -> ${3:ReturnType} {\n\t$0\n}",
            "if": "if ${1:condition} {\n\t$0\n}",
            "for": "for ${1:item} in ${2:iterable} {\n\t$0\n}",
            "while": "while ${1:condition} {\n\t$0\n}",
            "match": "match ${1:value} {\n\t${2:pattern} => $0\n}",
            "struct": "struct ${1:Name} {\n\t${2:field}: ${3:Type}\n}",
            "enum": "enum ${1:Name} {\n\t${2:Variant}\n}",
            "trait": "trait ${1:Name} {\n\tfn ${2:method}(self)$0\n}",
            "impl": "impl ${1:Type} {\n\t$0\n}",
            "class": "class ${1:Name} {\n\t$0\n}",
            "scene": "scene ${1:Name} {\n\t$0\n}",
            "let": "let ${1:name} = ${0:value}",
            "const": "const ${1:NAME}: ${2:Type} = ${0:value}",
            "import": "import ${1:module}",
            "use": "use ${1:module}",
        }

        for keyword, _token_type in KEYWORDS.items():
            # Skip non-keyword entries
            if keyword.startswith("_"):
                continue

            snippet = keyword_snippets.get(keyword)
            if snippet:
                completions.append(
                    types.CompletionItem(
                        label=keyword,
                        kind=types.CompletionItemKind.Keyword,
                        insert_text=snippet,
                        insert_text_format=types.InsertTextFormat.Snippet,
                        detail="keyword",
                        documentation=f"MathViz {keyword} keyword",
                    )
                )
            else:
                completions.append(
                    types.CompletionItem(
                        label=keyword,
                        kind=types.CompletionItemKind.Keyword,
                        detail="keyword",
                    )
                )

        self._keyword_completions = completions
        return completions

    def get_type_completions(self) -> list[types.CompletionItem]:
        """
        Get completion items for built-in types.

        Returns cached items after first call for performance.
        """
        if self._type_completions is not None:
            return self._type_completions

        completions: list[types.CompletionItem] = []

        for type_name, description in BUILTIN_TYPES.items():
            # Check if type is generic
            is_generic = type_name in ("List", "Set", "Dict", "Optional", "Result", "Tuple")

            if is_generic:
                snippet = f"{type_name}[${{1:T}}]"
                completions.append(
                    types.CompletionItem(
                        label=type_name,
                        kind=types.CompletionItemKind.TypeParameter,
                        insert_text=snippet,
                        insert_text_format=types.InsertTextFormat.Snippet,
                        detail="generic type",
                        documentation=description,
                    )
                )
            else:
                completions.append(
                    types.CompletionItem(
                        label=type_name,
                        kind=types.CompletionItemKind.TypeParameter,
                        detail="primitive type",
                        documentation=description,
                    )
                )

        self._type_completions = completions
        return completions

    def get_snippet_completions(self) -> list[types.CompletionItem]:
        """
        Get completion items for common code snippets.

        Returns cached items after first call for performance.
        """
        if self._snippet_completions is not None:
            return self._snippet_completions

        snippets = [
            (
                "main",
                "Main function",
                'fn main() {\n\t$0\n}',
            ),
            (
                "test",
                "Test function",
                '@test\nfn test_${1:name}() {\n\t$0\n}',
            ),
            (
                "forrange",
                "For loop with range",
                "for ${1:i} in range(${2:0}, ${3:n}) {\n\t$0\n}",
            ),
            (
                "foreach",
                "For-each loop",
                "for ${1:item} in ${2:collection} {\n\t$0\n}",
            ),
            (
                "ifel",
                "If-else statement",
                "if ${1:condition} {\n\t$2\n} else {\n\t$0\n}",
            ),
            (
                "ifelif",
                "If-elif-else statement",
                "if ${1:condition} {\n\t$2\n} elif ${3:condition} {\n\t$4\n} else {\n\t$0\n}",
            ),
            (
                "matchopt",
                "Match on Optional",
                "match ${1:value} {\n\tSome(${2:x}) => $3,\n\tNone => $0\n}",
            ),
            (
                "matchres",
                "Match on Result",
                "match ${1:value} {\n\tOk(${2:x}) => $3,\n\tErr(${4:e}) => $0\n}",
            ),
            (
                "impl",
                "Implementation block",
                "impl ${1:Type} {\n\tfn ${2:new}(${3:params}) -> Self {\n\t\t$0\n\t}\n}",
            ),
            (
                "impltrait",
                "Trait implementation",
                "impl ${1:Trait} for ${2:Type} {\n\t$0\n}",
            ),
            (
                "structnew",
                "Struct with constructor",
                "struct ${1:Name} {\n\t${2:field}: ${3:Type}\n}\n\nimpl $1 {\n\tfn new($2: $3) -> Self {\n\t\tSelf { $2 }\n\t}\n}",
            ),
            (
                "lambda",
                "Lambda expression",
                "|${1:x}| ${0:x * 2}",
            ),
            (
                "println",
                "Print with newline",
                'println("${1:message}")',
            ),
            (
                "printf",
                "Formatted print",
                'println(f"${1:message}: {${0:value}}")',
            ),
            (
                "scene",
                "Manim scene",
                "scene ${1:MyScene} {\n\tlet ${2:obj} = ${3:Circle}()\n\tplay(Create($2))\n\twait(1.0)\n}",
            ),
            (
                "animate",
                "Animation sequence",
                "play(${1:Transform}(${2:source}, ${3:target}))\nwait(${0:1.0})",
            ),
        ]

        completions = [
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Snippet,
                insert_text=snippet,
                insert_text_format=types.InsertTextFormat.Snippet,
                detail=f"snippet: {description}",
                documentation=f"Insert {description.lower()} snippet",
            )
            for name, description, snippet in snippets
        ]

        self._snippet_completions = completions
        return completions

    def get_stdlib_completions(self) -> list[types.CompletionItem]:
        """
        Get completion items for standard library functions.

        Returns cached items after first call for performance.
        """
        if self._stdlib_completions is not None:
            return self._stdlib_completions

        completions: list[types.CompletionItem] = []

        # Math functions
        for name, (signature, description) in STDLIB_MATH.items():
            completions.append(
                types.CompletionItem(
                    label=name,
                    kind=types.CompletionItemKind.Function,
                    detail=signature,
                    documentation=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=f"**{name}**\n\n{description}\n\n```mathviz\n{signature}\n```",
                    ),
                    insert_text=f"{name}($0)",
                    insert_text_format=types.InsertTextFormat.Snippet,
                )
            )

        # String functions
        for name, (signature, description) in STDLIB_STRING.items():
            if name not in [c.label for c in completions]:  # Avoid duplicates like 'len'
                completions.append(
                    types.CompletionItem(
                        label=name,
                        kind=types.CompletionItemKind.Function,
                        detail=signature,
                        documentation=types.MarkupContent(
                            kind=types.MarkupKind.Markdown,
                            value=f"**{name}**\n\n{description}\n\n```mathviz\n{signature}\n```",
                        ),
                        insert_text=f"{name}($0)",
                        insert_text_format=types.InsertTextFormat.Snippet,
                    )
                )

        # Collection functions
        for name, (signature, description) in STDLIB_COLLECTIONS.items():
            if name not in [c.label for c in completions]:  # Avoid duplicates
                completions.append(
                    types.CompletionItem(
                        label=name,
                        kind=types.CompletionItemKind.Function,
                        detail=signature,
                        documentation=types.MarkupContent(
                            kind=types.MarkupKind.Markdown,
                            value=f"**{name}**\n\n{description}\n\n```mathviz\n{signature}\n```",
                        ),
                        insert_text=f"{name}($0)",
                        insert_text_format=types.InsertTextFormat.Snippet,
                    )
                )

        self._stdlib_completions = completions
        return completions

    def get_symbol_completions(self, symbols: list[Symbol]) -> list[types.CompletionItem]:
        """
        Get completion items for local symbols.

        Args:
            symbols: List of symbols visible in the current scope

        Returns:
            List of completion items for the symbols
        """
        completions: list[types.CompletionItem] = []

        kind_map = {
            SymbolKind.FUNCTION: types.CompletionItemKind.Function,
            SymbolKind.VARIABLE: types.CompletionItemKind.Variable,
            SymbolKind.CONSTANT: types.CompletionItemKind.Constant,
            SymbolKind.PARAMETER: types.CompletionItemKind.Variable,
            SymbolKind.TYPE: types.CompletionItemKind.TypeParameter,
            SymbolKind.CLASS: types.CompletionItemKind.Class,
            SymbolKind.STRUCT: types.CompletionItemKind.Struct,
            SymbolKind.ENUM: types.CompletionItemKind.Enum,
            SymbolKind.ENUM_VARIANT: types.CompletionItemKind.EnumMember,
            SymbolKind.TRAIT: types.CompletionItemKind.Interface,
            SymbolKind.METHOD: types.CompletionItemKind.Method,
            SymbolKind.FIELD: types.CompletionItemKind.Field,
            SymbolKind.MODULE: types.CompletionItemKind.Module,
            SymbolKind.SCENE: types.CompletionItemKind.Class,
        }

        for symbol in symbols:
            kind = kind_map.get(symbol.kind, types.CompletionItemKind.Text)

            # Build documentation
            doc_parts: list[str] = []
            if symbol.signature:
                doc_parts.append(f"```mathviz\n{symbol.signature}\n```")
            if symbol.doc:
                doc_parts.append(symbol.doc)

            documentation = None
            if doc_parts:
                documentation = types.MarkupContent(
                    kind=types.MarkupKind.Markdown, value="\n\n".join(doc_parts)
                )

            # Build insert text for functions
            insert_text = symbol.name
            insert_text_format = types.InsertTextFormat.PlainText
            if symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                insert_text = f"{symbol.name}($0)"
                insert_text_format = types.InsertTextFormat.Snippet

            completions.append(
                types.CompletionItem(
                    label=symbol.name,
                    kind=kind,
                    detail=symbol.type_info or symbol.signature,
                    documentation=documentation,
                    insert_text=insert_text,
                    insert_text_format=insert_text_format,
                )
            )

        return completions

    def get_member_completions(self, base_type: str, symbols: list[Symbol]) -> list[types.CompletionItem]:
        """
        Get completion items for member access (after a dot).

        Args:
            base_type: The type of the expression before the dot
            symbols: List of potential member symbols

        Returns:
            List of completion items for members
        """
        # Filter to methods and fields of the base type
        members = [s for s in symbols if s.scope == base_type or s.kind in (SymbolKind.METHOD, SymbolKind.FIELD)]
        return self.get_symbol_completions(members)

    def get_all_completions(
        self, context: CompletionContext, local_symbols: list[Symbol]
    ) -> list[types.CompletionItem]:
        """
        Get all relevant completion items for a given context.

        Args:
            context: The completion context
            local_symbols: Symbols visible in the current scope

        Returns:
            Combined list of completion items
        """
        completions: list[types.CompletionItem] = []

        # Add appropriate completions based on context
        if context.is_in_type_position:
            # In type position, prioritize types
            completions.extend(self.get_type_completions())
        elif context.is_after_dot:
            # After dot, show member completions only
            # (Member completions would need type info to be accurate)
            pass
        elif context.is_in_import:
            # In import context, show modules
            pass
        else:
            # General context - show everything relevant
            if context.is_at_line_start:
                completions.extend(self.get_snippet_completions())

            completions.extend(self.get_keyword_completions())
            completions.extend(self.get_type_completions())
            completions.extend(self.get_stdlib_completions())
            completions.extend(self.get_symbol_completions(local_symbols))

        return completions
