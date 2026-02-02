# MathViz

A domain-specific language for mathematical animations.

MathViz extends Python with a clean, math-first syntax and compiles to pure Python
with first-class Manim integration. The standard file extension is `.mviz`.

Status: Alpha (work in progress).

## Highlights

- Unicode math operators (∈, ⊆, ∪, ∩, ∞, π, etc.)
- Clean block syntax with `let` and `fn`
- Manim-first `scene` blocks that compile to Python
- Static tooling: `check`, `analyze`, `typecheck`, `lint`, `fmt`
- Multi-file modules via `use` and `.mviz` imports
- REPL and watch mode for fast iteration
- Cross-platform desktop editor (Tauri + Monaco) in `editor/`

## Quick Start

### 1) Clone

```bash
git clone https://github.com/CyberSnakeH/MathViz.git
cd MathViz
```

### 2) Install

```bash
# Recommended (uv)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

Install with pipx (isolated CLI):

```bash
pipx install git+https://github.com/CyberSnakeH/MathViz.git
```

### 3) Run an example

```bash
mathviz run examples/hello_world.mviz
```

Or compile to Python:

```bash
mathviz compile examples/hello_world.mviz -o hello_world.py
python hello_world.py
```

## Language Example

```mathviz
let message = "Hello, MathViz!"
let x = 2 ^ 10

fn greet(name: String) -> String {
    return "Hello, " + name + "!"
}

print(message)
print(greet("World"))
print(x)  # 1024
```

### Manim Scene Example

```mathviz
from manim import Circle, Create, FadeOut

scene CircleAnimation {
    fn construct(self) {
        let circle = Circle()
        self.play(Create(circle))
        self.wait(1)
        self.play(FadeOut(circle))
    }
}
```

```bash
mathviz run examples/circle_animation.mviz --preview
```

## CLI (most used)

```bash
mathviz --help
mathviz compile file.mviz -o output.py
mathviz run file.mviz
mathviz check file.mviz
mathviz analyze file.mviz
mathviz typecheck file.mviz
mathviz fmt file.mviz
mathviz watch src/
```

## MathViz Editor (Tauri)

```bash
cd editor
npm ci
npm run tauri dev
```

Build a native app:

```bash
npm run tauri build
```

## Examples

- `examples/hello_world.mviz`
- `examples/set_operations.mviz`
- `examples/circle_animation.mviz`
- `examples/venn_diagram.mviz`
- `examples/multimodule/` (multi-file imports)

## Development

Requirements:
- Python 3.12+
- Node.js (for the editor)

Tests:

```bash
uv run pytest
```

Format and lint:

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run mypy src
```

Optional extras:

```bash
pip install -e ".[dev,watch,lsp]"
```

Run the language server:

```bash
mathviz-lsp
```

## Project Structure

```
.
├── src/            # MathViz compiler and runtime
├── editor/         # Tauri-based desktop editor
├── examples/       # Example .mviz programs
├── tests/          # Unit and integration tests
├── media/          # Assets used by examples
└── pyproject.toml  # Python package config
```

## License

MIT
