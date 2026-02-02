"""
MathViz Documentation Generator.

Generates documentation from doc comments in MathViz source files.
Supports Markdown and HTML output formats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import re

from mathviz.compiler.ast_nodes import (
    Program,
    FunctionDef,
    StructDef,
    TraitDef,
    EnumDef,
    ClassDef,
    ModuleDecl,
    ImplBlock,
    ConstDeclaration,
    BaseASTVisitor,
)


@dataclass
class DocItem:
    """A documented item."""
    name: str
    kind: str  # "function", "struct", "trait", "enum", "const", "module"
    doc: str
    signature: str
    location: Optional[str] = None
    children: List["DocItem"] = field(default_factory=list)
    params: List[Dict[str, str]] = field(default_factory=list)
    returns: Optional[str] = None
    examples: List[str] = field(default_factory=list)


@dataclass
class Documentation:
    """Complete documentation for a module."""
    name: str
    description: str
    items: List[DocItem] = field(default_factory=list)
    submodules: List["Documentation"] = field(default_factory=list)


class DocExtractor(BaseASTVisitor):
    """Extracts documentation from AST nodes."""

    def __init__(self) -> None:
        self.items: List[DocItem] = []
        self._current_module: Optional[str] = None

    def extract(self, program: Program) -> List[DocItem]:
        """Extract all documented items from a program."""
        self.items = []
        self.visit(program)
        return self.items

    def _parse_doc_comment(self, doc: Optional[str]) -> Dict[str, Any]:
        """Parse a doc comment into structured data."""
        if not doc:
            return {"description": "", "params": [], "returns": None, "examples": []}

        lines = doc.strip().split('\n')
        description_lines = []
        params = []
        returns = None
        examples = []
        in_example = False
        example_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("@param"):
                match = re.match(r"@param\s+(\w+)\s*:?\s*(.*)", line)
                if match:
                    params.append({"name": match.group(1), "desc": match.group(2)})
            elif line.startswith("@returns") or line.startswith("@return"):
                match = re.match(r"@returns?\s*:?\s*(.*)", line)
                if match:
                    returns = match.group(1)
            elif line.startswith("@example"):
                in_example = True
            elif line.startswith("@"):
                in_example = False
                if example_lines:
                    examples.append('\n'.join(example_lines))
                    example_lines = []
            elif in_example:
                example_lines.append(line)
            else:
                description_lines.append(line)

        if example_lines:
            examples.append('\n'.join(example_lines))

        return {
            "description": '\n'.join(description_lines).strip(),
            "params": params,
            "returns": returns,
            "examples": examples,
        }

    def visit_function_def(self, node: FunctionDef) -> None:
        """Extract documentation from a function."""
        doc_str = node.doc_comment.content if node.doc_comment else ""
        parsed = self._parse_doc_comment(doc_str)

        # Build signature
        params = []
        for p in node.parameters:
            param_str = p.name
            if p.type_annotation:
                param_str += f": {self._type_to_str(p.type_annotation)}"
            params.append(param_str)

        ret_type = ""
        if node.return_type:
            ret_type = f" -> {self._type_to_str(node.return_type)}"

        signature = f"fn {node.name}({', '.join(params)}){ret_type}"

        self.items.append(DocItem(
            name=node.name,
            kind="function",
            doc=parsed["description"],
            signature=signature,
            params=parsed["params"],
            returns=parsed["returns"],
            examples=parsed["examples"],
        ))

    def visit_struct_def(self, node: StructDef) -> None:
        """Extract documentation from a struct."""
        doc_str = node.doc_comment.content if node.doc_comment else ""
        parsed = self._parse_doc_comment(doc_str)

        # Build signature with fields
        fields = []
        for f in node.fields:
            field_str = f.name
            if f.type_annotation:
                field_str += f": {self._type_to_str(f.type_annotation)}"
            fields.append(field_str)

        type_params = ""
        if node.type_params:
            type_params = f"<{', '.join(tp.name for tp in node.type_params)}>"

        signature = f"struct {node.name}{type_params} {{ {', '.join(fields)} }}"

        self.items.append(DocItem(
            name=node.name,
            kind="struct",
            doc=parsed["description"],
            signature=signature,
            examples=parsed["examples"],
        ))

    def visit_trait_def(self, node: TraitDef) -> None:
        """Extract documentation from a trait."""
        doc_str = node.doc_comment.content if node.doc_comment else ""
        parsed = self._parse_doc_comment(doc_str)

        type_params = ""
        if node.type_params:
            type_params = f"<{', '.join(tp.name for tp in node.type_params)}>"

        signature = f"trait {node.name}{type_params}"

        self.items.append(DocItem(
            name=node.name,
            kind="trait",
            doc=parsed["description"],
            signature=signature,
            examples=parsed["examples"],
        ))

    def visit_enum_def(self, node: EnumDef) -> None:
        """Extract documentation from an enum."""
        doc_str = node.doc_comment.content if node.doc_comment else ""
        parsed = self._parse_doc_comment(doc_str)

        variants = [v.name for v in node.variants]
        type_params = ""
        if node.type_params:
            type_params = f"<{', '.join(tp.name for tp in node.type_params)}>"

        signature = f"enum {node.name}{type_params} {{ {', '.join(variants)} }}"

        self.items.append(DocItem(
            name=node.name,
            kind="enum",
            doc=parsed["description"],
            signature=signature,
            examples=parsed["examples"],
        ))

    def visit_const_declaration(self, node: ConstDeclaration) -> None:
        """Extract documentation from a constant."""
        signature = f"const {node.name}"
        if node.type_annotation:
            signature += f": {self._type_to_str(node.type_annotation)}"

        self.items.append(DocItem(
            name=node.name,
            kind="const",
            doc="",
            signature=signature,
        ))

    def _type_to_str(self, type_ann) -> str:
        """Convert type annotation to string."""
        from mathviz.compiler.ast_nodes import SimpleType, GenericType
        if isinstance(type_ann, SimpleType):
            return type_ann.name
        elif isinstance(type_ann, GenericType):
            args = ", ".join(self._type_to_str(a) for a in type_ann.type_args)
            return f"{type_ann.base}[{args}]"
        return str(type_ann)


class MarkdownGenerator:
    """Generates Markdown documentation."""

    def generate(self, items: List[DocItem], title: str = "API Documentation") -> str:
        """Generate Markdown documentation."""
        lines = [
            f"# {title}",
            "",
        ]

        # Group by kind
        groups = {}
        for item in items:
            if item.kind not in groups:
                groups[item.kind] = []
            groups[item.kind].append(item)

        order = ["const", "function", "struct", "trait", "enum"]
        kind_titles = {
            "const": "Constants",
            "function": "Functions",
            "struct": "Structs",
            "trait": "Traits",
            "enum": "Enums",
        }

        for kind in order:
            if kind in groups:
                lines.append(f"## {kind_titles.get(kind, kind.title())}")
                lines.append("")
                for item in groups[kind]:
                    lines.extend(self._render_item(item))
                lines.append("")

        return '\n'.join(lines)

    def _render_item(self, item: DocItem) -> List[str]:
        """Render a single documented item."""
        lines = [
            f"### `{item.name}`",
            "",
            "```mathviz",
            item.signature,
            "```",
            "",
        ]

        if item.doc:
            lines.append(item.doc)
            lines.append("")

        if item.params:
            lines.append("**Parameters:**")
            lines.append("")
            for p in item.params:
                lines.append(f"- `{p['name']}`: {p['desc']}")
            lines.append("")

        if item.returns:
            lines.append(f"**Returns:** {item.returns}")
            lines.append("")

        if item.examples:
            lines.append("**Examples:**")
            lines.append("")
            for ex in item.examples:
                lines.append("```mathviz")
                lines.append(ex)
                lines.append("```")
            lines.append("")

        return lines


class HTMLGenerator:
    """Generates HTML documentation."""

    def generate(self, items: List[DocItem], title: str = "API Documentation") -> str:
        """Generate HTML documentation."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "<style>",
            self._get_styles(),
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
        ]

        # Group by kind
        groups = {}
        for item in items:
            if item.kind not in groups:
                groups[item.kind] = []
            groups[item.kind].append(item)

        order = ["const", "function", "struct", "trait", "enum"]
        kind_titles = {
            "const": "Constants",
            "function": "Functions",
            "struct": "Structs",
            "trait": "Traits",
            "enum": "Enums",
        }

        for kind in order:
            if kind in groups:
                html.append(f"<h2>{kind_titles.get(kind, kind.title())}</h2>")
                for item in groups[kind]:
                    html.extend(self._render_item(item))

        html.extend([
            "</body>",
            "</html>",
        ])

        return '\n'.join(html)

    def _get_styles(self) -> str:
        """Get CSS styles."""
        return """
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #eee; }
        h2 { color: #555; margin-top: 2em; }
        h3 { color: #666; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
        code { font-family: 'Fira Code', monospace; }
        .params { margin-left: 20px; }
        .param-name { font-weight: bold; color: #0066cc; }
        .returns { color: #009900; }
        .example { background: #f0f8ff; padding: 10px; border-left: 3px solid #0066cc; }
        """

    def _render_item(self, item: DocItem) -> List[str]:
        """Render a single documented item."""
        lines = [
            f'<div class="doc-item">',
            f'<h3><code>{item.name}</code></h3>',
            f'<pre><code>{self._escape(item.signature)}</code></pre>',
        ]

        if item.doc:
            lines.append(f'<p>{self._escape(item.doc)}</p>')

        if item.params:
            lines.append('<div class="params"><strong>Parameters:</strong><ul>')
            for p in item.params:
                lines.append(f'<li><span class="param-name">{p["name"]}</span>: {self._escape(p["desc"])}</li>')
            lines.append('</ul></div>')

        if item.returns:
            lines.append(f'<p class="returns"><strong>Returns:</strong> {self._escape(item.returns)}</p>')

        if item.examples:
            lines.append('<div class="examples"><strong>Examples:</strong>')
            for ex in item.examples:
                lines.append(f'<pre class="example"><code>{self._escape(ex)}</code></pre>')
            lines.append('</div>')

        lines.append('</div>')
        return lines

    def _escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))


def generate_docs(program: Program, format: str = "markdown", title: str = "API Documentation") -> str:
    """
    Generate documentation from a MathViz program.

    Args:
        program: The parsed AST
        format: Output format ("markdown" or "html")
        title: Document title

    Returns:
        Generated documentation as a string
    """
    extractor = DocExtractor()
    items = extractor.extract(program)

    if format == "html":
        generator = HTMLGenerator()
    else:
        generator = MarkdownGenerator()

    return generator.generate(items, title)
