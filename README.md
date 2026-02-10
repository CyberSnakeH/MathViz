<p align="center">
  <img src="docs/assets/logo.svg" alt="MathViz Logo" width="120" height="120">
</p>

<h1 align="center">MathViz</h1>

<p align="center">
  <strong>A domain-specific language for mathematical animations</strong>
</p>

<p align="center">
  <a href="https://github.com/CyberSnakeH/MathViz/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CyberSnakeH/MathViz/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/mathviz/"><img src="https://img.shields.io/pypi/v/mathviz?style=flat-square&color=7aa2f7" alt="PyPI"></a>
  <a href="https://github.com/CyberSnakeH/MathViz/releases"><img src="https://img.shields.io/github/v/release/CyberSnakeH/MathViz?style=flat-square&color=7aa2f7" alt="Release"></a>
  <a href="https://github.com/CyberSnakeH/MathViz/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CyberSnakeH/MathViz?style=flat-square&color=9ece6a" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12+-bb9af7?style=flat-square" alt="Python"></a>
  <a href="https://github.com/CyberSnakeH/MathViz/stargazers"><img src="https://img.shields.io/github/stars/CyberSnakeH/MathViz?style=flat-square&color=e0af68" alt="Stars"></a>
  <a href="https://cybersnakeh.github.io/MathViz"><img src="https://img.shields.io/badge/docs-website-7dcfff?style=flat-square" alt="Website"></a>
</p>

<p align="center">
  <a href="https://cybersnakeh.github.io/MathViz">Website</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#editor">Editor</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

MathViz is a modern programming language designed specifically for creating mathematical animations. It extends Python with a clean, math-first syntax and compiles to pure Python with first-class [Manim](https://www.manim.community/) integration.

## Features

- **Unicode Math Operators** — Write math naturally: `∈`, `⊆`, `∪`, `∩`, `∞`, `π`, `≤`, `≥`, `≠`
- **Clean Syntax** — Modern block syntax with `let`, `fn`, `scene`, and pattern matching
- **Manim Integration** — First-class support for creating mathematical animations
- **Static Analysis** — Built-in `check`, `analyze`, `typecheck`, `lint`, and `fmt` commands
- **Module System** — Multi-file projects with `use` imports
- **Desktop Editor** — Cross-platform editor with live preview (Tauri + React)
- **VS Code Extension** — Syntax highlighting, snippets, and bracket matching for VS Code
- **Auto-update** — Automatic update detection with PyPI version check
- **Fast Iteration** — REPL mode and file watcher for rapid development

## Installation

### Using pipx (Recommended)

```bash
pipx install mathviz
```

### Using pip

```bash
pip install mathviz
```

### From source

```bash
git clone https://github.com/CyberSnakeH/MathViz.git
cd MathViz
uv sync --dev
```

### Prerequisites

- Python 3.12 or higher
- [Manim](https://docs.manim.community/en/stable/installation.html) (for animations)
- [LaTeX](https://www.latex-project.org/get/) (optional, for math rendering)

## Quick Start

### Hello World

Create a file `hello.mviz`:

```mviz
fn main() {
    println("Hello, MathViz!")
}
```

Run it:

```bash
mathviz exec hello.mviz
```

### Your First Animation

Create `animation.mviz`:

```mviz
from manim import Circle, Create, FadeOut

scene CircleAnimation {
    fn construct(self) {
        let circle = Circle()
        circle.set_color(BLUE)

        self.play(Create(circle))
        self.wait(1)
        self.play(FadeOut(circle))
    }
}
```

Run with preview:

```bash
mathviz run animation.mviz --preview
```

### Language Features

```mviz
// Variables with type inference
let x = 42
let name = "MathViz"
let pi = 3.14159

// Functions with type annotations
fn add(a: int, b: int) -> int {
    return a + b
}

// Pattern matching
let result = match x {
    0 -> "zero"
    1 -> "one"
    _ -> "other"
}

// For loops with ranges
for i in 0..10 {
    println(i)
}

// List comprehensions
let squares = [x^2 for x in 1..=10]

// Unicode math
let is_member = 5 ∈ {1, 2, 3, 4, 5}
let union = {1, 2} ∪ {3, 4}
```

## CLI Reference

```bash
# Compile to Python
mathviz compile file.mviz -o output.py

# Run with Manim (animations)
mathviz run file.mviz --preview

# Execute script (no Manim)
mathviz exec file.mviz

# Static analysis
mathviz check file.mviz      # Syntax check
mathviz typecheck file.mviz  # Type check
mathviz analyze file.mviz    # Full analysis
mathviz lint file.mviz       # Linter

# Formatting
mathviz fmt file.mviz

# Development
mathviz watch src/           # Watch mode
mathviz repl                 # Interactive REPL

# Project management
mathviz new my-project       # Create new project
mathviz build                # Build project
mathviz test                 # Run tests
```

## VS Code Extension

Install the MathViz extension for VS Code with syntax highlighting, 18 code snippets, and smart bracket matching.

```bash
cd vscode-mathviz
npx @vscode/vsce package
code --install-extension mathviz-0.1.0.vsix
```

Open any `.mviz` or `.mvz` file and the extension activates automatically.

## Editor

MathViz includes a cross-platform desktop editor built with Tauri and React.

### Features

- Syntax highlighting for `.mviz` files
- Live animation preview
- Integrated terminal
- File explorer
- Command palette (`Ctrl+P`)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F5` | Run with Manim |
| `F6` | Execute script |
| `F8` | Compile only |
| `Ctrl+S` | Save |
| `Ctrl+P` | Command palette |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+J` | Toggle terminal |

### Running the Editor

```bash
cd editor
npm install
npm run tauri dev
```

### Building

```bash
npm run tauri build
```

Binaries will be in `editor/src-tauri/target/release/bundle/`.

## Project Structure

```
MathViz/
├── src/mathviz/          # Compiler and runtime
│   ├── compiler/         # Lexer, parser, codegen
│   ├── utils/            # Errors, diagnostics
│   └── cli.py            # CLI entry point
├── editor/               # Desktop editor (Tauri)
│   ├── src/              # React frontend
│   └── src-tauri/        # Rust backend
├── vscode-mathviz/       # VS Code extension
│   ├── syntaxes/         # TextMate grammar
│   └── snippets/         # Code snippets
├── examples/             # Example programs
├── tests/                # Test suite
└── docs/                 # Documentation
```

## Examples

Explore the `examples/` directory:

- [`hello_world.mviz`](examples/hello_world.mviz) — Basic syntax
- [`circle_animation.mviz`](examples/circle_animation.mviz) — Simple animation
- [`set_operations.mviz`](examples/set_operations.mviz) — Unicode math
- [`venn_diagram.mviz`](examples/venn_diagram.mviz) — Complex animation
- [`multimodule/`](examples/multimodule/) — Multi-file project

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format
uv run ruff format src tests

# Lint
uv run ruff check src tests

# Type check
uv run mypy src
```

### Language Server

```bash
pip install -e ".[lsp]"
mathviz-lsp
```

## Contributing

Contributions are welcome! Please read our **[Contributing Guide](CONTRIBUTING.md)** for details on:

- Development environment setup
- Code style and conventions
- Commit message guidelines
- Testing requirements
- Pull request process

Quick start:

```bash
# Fork and clone
git clone https://github.com/CyberSnakeH/MathViz.git
cd MathViz

# Create a branch
git checkout -b feature/my-feature

# Make changes, then submit a PR
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Manim Community](https://www.manim.community/) — The animation engine
- [Tauri](https://tauri.app/) — Desktop app framework
- [Monaco Editor](https://microsoft.github.io/monaco-editor/) — Code editor

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/CyberSnakeH">CyberSnakeH</a>
</p>
