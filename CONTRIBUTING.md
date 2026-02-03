# Contributing to MathViz

First off, thank you for considering contributing to MathViz! It's people like you that make MathViz such a great tool for the mathematical visualization community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment](#development-environment)
  - [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Guidelines](#commit-guidelines)
  - [Code Style](#code-style)
  - [Testing](#testing)
- [Architecture Overview](#architecture-overview)
  - [Compiler Pipeline](#compiler-pipeline)
  - [Editor Architecture](#editor-architecture)
- [Review Process](#review-process)
- [Release Process](#release-process)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. By participating, you are expected to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Development Environment

#### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Compiler runtime |
| Node.js | 18+ | Editor frontend |
| Rust | 1.70+ | Editor backend (Tauri) |
| uv | latest | Python package management |
| Git | 2.0+ | Version control |

#### Quick Setup

```bash
# Clone the repository
git clone https://github.com/CyberSnakeH/MathViz.git
cd MathViz

# Install Python dependencies
uv sync --dev

# Verify installation
uv run mathviz --version
uv run pytest

# (Optional) Set up the editor
cd editor
npm install
npm run tauri dev
```

#### IDE Setup

**VS Code** (Recommended):
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.enable": true
}
```

**Recommended Extensions**:
- Python (ms-python.python)
- Ruff (charliermarsh.ruff)
- Rust Analyzer (rust-lang.rust-analyzer)
- Tauri (tauri-apps.tauri-vscode)

### Project Structure

```
MathViz/
├── src/mathviz/              # 🐍 Python compiler
│   ├── compiler/             # Core compilation pipeline
│   │   ├── lexer.py          # Tokenization
│   │   ├── parser.py         # AST generation
│   │   ├── type_checker.py   # Static type analysis
│   │   ├── codegen.py        # Python code generation
│   │   └── __init__.py       # CompilationPipeline
│   ├── utils/                # Utilities
│   │   ├── errors.py         # Error types
│   │   └── diagnostics.py    # Diagnostic codes
│   └── cli.py                # Command-line interface
│
├── editor/                   # 🖥️ Desktop editor
│   ├── src/                  # React frontend
│   │   ├── components/       # UI components
│   │   ├── stores/           # Zustand state
│   │   └── App.tsx           # Main application
│   └── src-tauri/            # Rust backend
│       └── src/
│           ├── commands/     # Tauri commands
│           └── main.rs       # Entry point
│
├── tests/                    # 🧪 Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test data
│
├── examples/                 # 📚 Example programs
├── docs/                     # 📖 Documentation
└── pyproject.toml           # Project configuration
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**Great bug reports include:**

1. **Summary**: Clear, descriptive title
2. **Environment**: OS, Python version, MathViz version
3. **Steps to Reproduce**: Minimal code example
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Error Output**: Full error message/traceback

**Template:**

```markdown
### Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.12.1]
- MathViz: [e.g., 0.1.6]

### Description
[Clear description of the bug]

### Steps to Reproduce
1. Create file `test.mviz`:
```mviz
// minimal code that reproduces the issue
```
2. Run `mathviz compile test.mviz`
3. See error

### Expected Behavior
[What you expected to happen]

### Actual Behavior
[What actually happened]

### Error Output
```
[paste full error here]
```
```

### Suggesting Features

Feature requests are welcome! Please provide:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches you've thought about
4. **Code Examples**: How would users use this feature?

### Pull Requests

We actively welcome pull requests! Here's the process:

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Implement** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## Development Workflow

### Branching Strategy

```
main                    # Stable, release-ready
├── feature/xyz         # New features
├── fix/issue-123       # Bug fixes
├── refactor/xyz        # Code improvements
└── docs/xyz            # Documentation
```

**Branch Naming Convention:**

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/pattern-matching` |
| Bug Fix | `fix/issue-number` | `fix/issue-42` |
| Refactor | `refactor/description` | `refactor/lexer-performance` |
| Docs | `docs/description` | `docs/api-reference` |

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |

**Examples:**

```bash
# Feature
feat(parser): add support for pattern matching

# Bug fix
fix(type-checker): resolve enum type recognition issue

# Documentation
docs(readme): update installation instructions

# Refactoring
refactor(lexer): improve token scanning performance
```

**Commit Message Best Practices:**

- Use imperative mood ("add" not "added")
- Keep the subject line under 72 characters
- Reference issues when applicable: `fix(parser): handle edge case (#123)`
- Explain *why* in the body, not *what* (the code shows what)

### Code Style

#### Python

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
uv run ruff format src tests

# Check for issues
uv run ruff check src tests

# Auto-fix issues
uv run ruff check --fix src tests

# Type checking
uv run mypy src
```

**Style Guidelines:**

```python
# ✅ Good: Clear, documented, typed
def compile_source(
    source: str,
    *,
    optimize: bool = True,
    debug: bool = False,
) -> CompilationResult:
    """Compile MathViz source code to Python.

    Args:
        source: The MathViz source code to compile.
        optimize: Enable optimization passes.
        debug: Include debug information.

    Returns:
        CompilationResult containing the generated code or errors.

    Raises:
        SyntaxError: If the source contains invalid syntax.
    """
    ...

# ❌ Bad: Unclear, untyped, undocumented
def compile(s, opt=True, dbg=False):
    ...
```

#### TypeScript/React

We follow standard React conventions:

```bash
cd editor
npm run lint
npm run format
```

**Style Guidelines:**

```typescript
// ✅ Good: Typed, clear naming, documented
interface CompilerState {
  status: 'idle' | 'compiling' | 'running' | 'success' | 'error';
  output: string[];
  lastResult: CompileResult | null;
}

const useCompilerStore = create<CompilerState>()((set) => ({
  // ...
}));

// ❌ Bad: Untyped, unclear
const store = create((set) => ({
  s: 'idle',
  o: [],
  r: null,
}));
```

#### Rust

Follow standard Rust conventions with `cargo fmt` and `cargo clippy`:

```bash
cd editor/src-tauri
cargo fmt
cargo clippy
```

### Testing

#### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/unit/test_lexer.py

# With coverage
uv run pytest --cov=mathviz --cov-report=html

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

#### Writing Tests

**Test File Structure:**

```python
# tests/unit/test_parser.py

import pytest
from mathviz.compiler.parser import Parser
from mathviz.compiler.lexer import Lexer


class TestParser:
    """Tests for the MathViz parser."""

    def test_parse_let_statement(self):
        """Parser correctly handles let statements."""
        source = "let x = 42"
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())

        ast = parser.parse()

        assert len(ast.statements) == 1
        assert ast.statements[0].name == "x"
        assert ast.statements[0].value.value == 42

    def test_parse_function_definition(self):
        """Parser correctly handles function definitions."""
        source = """
        fn add(a: int, b: int) -> int {
            return a + b
        }
        """
        # ...

    @pytest.mark.parametrize("source,expected", [
        ("1 + 2", 3),
        ("10 - 5", 5),
        ("3 * 4", 12),
    ])
    def test_arithmetic_expressions(self, source, expected):
        """Parser handles arithmetic expressions."""
        # ...
```

**Test Categories:**

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/unit/` | Test individual functions/classes |
| Integration | `tests/integration/` | Test component interactions |
| E2E | `tests/e2e/` | Test full compilation pipeline |

## Architecture Overview

### Compiler Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Source Code (.mviz)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LEXER (lexer.py)                                           │
│  - Character stream → Token stream                          │
│  - Handles Unicode operators (∈, ∪, ∩, etc.)               │
│  - Tracks source locations for error reporting              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PARSER (parser.py)                                         │
│  - Token stream → Abstract Syntax Tree                      │
│  - Recursive descent parsing                                │
│  - Operator precedence climbing                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  TYPE CHECKER (type_checker.py)                             │
│  - Type inference and validation                            │
│  - Symbol table management                                  │
│  - Error detection                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ANALYZERS (analyzers/)                                     │
│  - Linter: Style and best practices                         │
│  - Semantic analysis: Variable usage, dead code             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  CODE GENERATOR (codegen.py)                                │
│  - AST → Python source code                                 │
│  - Manim integration                                        │
│  - Optimization passes                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Python Code (.py)                        │
└─────────────────────────────────────────────────────────────┘
```

### Editor Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TypeScript)             │
├─────────────────────────────────────────────────────────────┤
│  Components:                                                │
│  ├── Editor (Monaco)      - Code editing                    │
│  ├── FileTree             - Project navigation              │
│  ├── Terminal             - Integrated terminal             │
│  ├── Preview              - Animation preview               │
│  └── DebugPanel           - Run & compile controls          │
│                                                             │
│  State Management (Zustand):                                │
│  ├── editorStore          - Open files, active tab          │
│  ├── compilerStore        - Compilation state               │
│  ├── fileStore            - File tree state                 │
│  └── layoutStore          - UI layout state                 │
└─────────────────────────────────────────────────────────────┘
                              │
                         Tauri IPC
                              │
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (Rust + Tauri)                    │
├─────────────────────────────────────────────────────────────┤
│  Commands:                                                  │
│  ├── file::*              - File operations                 │
│  ├── compiler::*          - MathViz compilation             │
│  ├── terminal::*          - PTY management                  │
│  └── git::*               - Git operations                  │
└─────────────────────────────────────────────────────────────┘
```

## Review Process

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows the style guidelines
- [ ] Tests pass locally (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check src tests`)
- [ ] Type checking passes (`uv run mypy src`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

### Review Criteria

Reviewers will check:

1. **Correctness**: Does the code do what it should?
2. **Tests**: Are there adequate tests?
3. **Performance**: Any performance implications?
4. **Security**: Any security concerns?
5. **Style**: Does it follow our conventions?
6. **Documentation**: Is it properly documented?

### Merge Requirements

- At least 1 approving review
- All CI checks passing
- No unresolved conversations
- Up-to-date with `main` branch

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR: Breaking changes
MINOR: New features (backward compatible)
PATCH: Bug fixes (backward compatible)
```

### Release Checklist

1. Update version in:
   - `pyproject.toml`
   - `editor/package.json`
   - `editor/src-tauri/Cargo.toml`
   - `editor/src-tauri/tauri.conf.json`

2. Update CHANGELOG.md

3. Create release commit:
   ```bash
   git commit -m "chore(release): v0.1.7"
   ```

4. Create and push tag:
   ```bash
   git tag -a v0.1.7 -m "MathViz v0.1.7"
   git push origin main --tags
   ```

5. Create GitHub release with changelog

---

## Questions?

- **GitHub Issues**: [github.com/CyberSnakeH/MathViz/issues](https://github.com/CyberSnakeH/MathViz/issues)
- **Documentation**: [cybersnakeh.github.io/MathViz](https://cybersnakeh.github.io/MathViz)

Thank you for contributing to MathViz! 🎉
