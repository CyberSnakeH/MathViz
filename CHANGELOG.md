# Changelog

All notable changes to MathViz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

---

## [0.1.6] - 2025-02-03

### Added
- Keyboard shortcuts in editor: F5 (Run Manim), F6 (Execute script), F8 (Compile)
- `exec_mathviz` Tauri command for running simple scripts without Manim
- Professional README with badges and logo
- Project logo (`docs/assets/logo.svg`)
- MIT License file
- Comprehensive CONTRIBUTING.md guide
- GitHub Actions CI/CD workflow
- Issue and PR templates

### Fixed
- `mathviz exec` command now correctly shows print output (`__name__` set to `"__main__"`)
- Type checker now recognizes enum, struct, and class types as valid identifiers
- Serde camelCase serialization for Rust ↔ TypeScript communication

### Changed
- Updated command palette with new run/execute/compile options
- Welcome screen now shows F5, F6, F8 shortcuts

---

## [0.1.5] - 2025-01-28

### Added
- Pattern matching with `match` expressions
- Tuple destructuring in variable declarations
- For loop destructuring (`for (a, b) in items`)
- `loop` and `break`/`continue` keywords
- Range expressions (`0..10`, `0..=10`)

### Fixed
- Type checker for tuple destructuring in for loops

---

## [0.1.4] - 2025-01-20

### Added
- Tauri desktop editor with Monaco integration
- File explorer with tree view
- Integrated terminal (PTY)
- Live Manim preview panel
- Git integration panel
- Tokyo Night theme

### Changed
- Improved error messages with source locations

---

## [0.1.3] - 2025-01-15

### Added
- `scene` blocks for Manim animations
- `from` imports for Python modules
- Method chaining support
- `self` parameter in scene methods

### Fixed
- Code generation for nested function calls

---

## [0.1.2] - 2025-01-10

### Added
- Type annotations for functions
- Return type inference
- Basic type checking
- `analyze` command for comprehensive analysis

### Changed
- Improved parser error recovery

---

## [0.1.1] - 2025-01-05

### Added
- `fmt` command for code formatting
- `lint` command for style checking
- `watch` command for file watching
- REPL mode (`mathviz repl`)

### Fixed
- Unicode operator parsing

---

## [0.1.0] - 2025-01-01

### Added
- Initial release
- MathViz language specification
- Lexer with Unicode support (∈, ⊆, ∪, ∩, ∞, π)
- Recursive descent parser
- Python code generation
- `compile` command
- `run` command with Manim integration
- `check` command for syntax validation
- Basic examples

---

[Unreleased]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/CyberSnakeH/MathViz/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CyberSnakeH/MathViz/releases/tag/v0.1.0
