# MathViz Language Support

Syntax highlighting, snippets, and language configuration for the [MathViz](https://github.com/cybersnakeh/MathViz) DSL.

MathViz is a domain-specific language that compiles to Python/Manim for creating mathematical visualizations.

## Features

- Full syntax highlighting for `.mviz` and `.mvz` files
- Smart bracket matching and auto-closing
- Code snippets for common patterns
- Unicode math operator support
- Manim class and function recognition
- Tokyo Night theme color customizations

## Supported Syntax

- **Keywords**: `let`, `mut`, `const`, `fn`, `scene`, `class`, `struct`, `trait`, `impl`, `enum`, `mod`
- **Control flow**: `if`, `else`, `elif`, `for`, `while`, `break`, `continue`, `return`, `match`, `loop`
- **Operators**: `->`, `=>`, `..`, `..=`, `::`, `|>`, and Unicode math operators (`∈`, `∉`, `⊆`, `∪`, `∩`, etc.)
- **Built-in types**: `Int`, `Float`, `Bool`, `String`, `List`, `Set`, `Dict`, `Vec`, `Mat`, `Matrix`
- **Manim objects**: `Circle`, `Square`, `Line`, `Arrow`, `Text`, `MathTex`, `Axes`, `NumberPlane`, etc.
- **Animations**: `Create`, `FadeIn`, `FadeOut`, `Write`, `Transform`, `ReplacementTransform`, etc.
- **Math functions**: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `matmul`, `dot`, `transpose`, etc.
- **Decorators**: `@jit`, `@njit`, `@vectorize`, `@parallel`
- **Comments**: `//`, `#`, `/* */`, `///` (doc comments)
- **String interpolation**: `"Hello {name}"`

## Snippets

| Prefix   | Description              |
|----------|--------------------------|
| `scene`  | Scene block              |
| `fn`     | Function definition      |
| `pub fn` | Public function          |
| `for`    | For loop                 |
| `if`     | If block                 |
| `ife`    | If-else block            |
| `match`  | Match expression         |
| `let`    | Variable declaration     |
| `letm`   | Mutable variable         |
| `const`  | Constant declaration     |
| `use`    | Module import            |
| `comp`   | List comprehension       |
| `play`   | Play animation           |
| `class`  | Class definition         |
| `struct` | Struct definition        |
| `impl`   | Implementation block     |
| `while`  | While loop               |
| `try`    | Try-catch block          |

## Installation

### From VSIX

```bash
code --install-extension mathviz-0.1.0.vsix
```

### From Source

```bash
cd vscode-mathviz
npx vsce package
code --install-extension mathviz-0.1.0.vsix
```

## Example

```mathviz
use math::{sin, cos, PI}

scene SinWave {
    let axes = Axes(x_range: [-PI, PI], y_range: [-1.5, 1.5])
    let curve = axes.plot(|x| sin(x), color: BLUE)

    self.play(Create(axes))
    self.play(Create(curve))
}
```

## License

MIT
