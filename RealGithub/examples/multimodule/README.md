# MathViz Multi-Module Example

This example demonstrates the MathViz module system, allowing you to organize
your code across multiple `.mviz` files.

## Structure

```
multimodule/
├── main.mviz          # Main entry point
├── geometry.mviz      # Geometry calculations (distance, area, etc.)
├── colors.mviz        # Color definitions and utilities
├── animations.mviz    # Animation utilities (uses geometry and colors)
└── README.md          # This file
```

## Module System Features

### Importing Modules

Use the `use` statement to import other `.mviz` files:

```mviz
use geometry        // Imports ./geometry.mviz
use lib.math        // Imports ./lib/math.mviz
use utils           // Can import ./utils.mviz or ./utils/mod.mviz
```

### Accessing Module Members

Access module members using dot notation:

```mviz
use geometry

fn main() {
    let d = geometry.distance(0, 0, 3, 4)
}
```

### Mixing with Python Imports

Python modules (numpy, manim, etc.) are automatically recognized:

```mviz
use numpy      // Standard Python import
use geometry   // MathViz module import

fn main() {
    let arr = numpy.array([1, 2, 3])
    let d = geometry.distance(0, 0, 3, 4)
}
```

## Running the Example

```bash
# Compile main.mviz
mathviz compile main.mviz -o output.py

# Or run directly
mathviz run main.mviz
```

## Generated Code

MathViz modules are compiled into Python classes with static methods:

```python
# Generated from geometry.mviz
class geometry:
    @staticmethod
    def distance(x1, y1, x2, y2):
        dx = (x2 - x1)
        dy = (y2 - y1)
        return np.sqrt(((dx * dx) + (dy * dy)))

    @staticmethod
    def midpoint(x1, y1, x2, y2):
        return (((x1 + x2) / 2), ((y1 + y2) / 2))
```

## Dependency Resolution

The module loader handles:
- **Relative imports**: `./module.mviz` relative to current file
- **Directory modules**: `./module/mod.mviz`
- **Nested imports**: `./lib/sub/module.mviz`
- **Search paths**: Additional library directories
- **Circular dependency detection**: Prevents infinite loops
- **Topological ordering**: Compiles dependencies first
