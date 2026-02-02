"""
MathViz Module Loader.

Handles resolution and loading of MathViz modules (.mviz files).

The module system supports:
- Importing other .mviz files with `use module`
- Nested modules with `use lib.math`
- Directory modules with `use utils` -> utils/mod.mviz
- Visibility control with `pub` keyword
- Circular dependency detection

Example:
    // geometry.mviz
    pub fn distance(x1, y1, x2, y2) {
        return sqrt((x2-x1)^2 + (y2-y1)^2)
    }

    // main.mviz
    use geometry
    let d = geometry.distance(0, 0, 3, 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mathviz.compiler.ast_nodes import (
        FunctionDef,
        ModuleDecl,
        Program,
        UseStatement,
    )

from mathviz.utils.errors import SourceLocation


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ModuleInfo:
    """
    Information about a loaded MathViz module.

    Attributes:
        name: Qualified module name (e.g., "mylib.geometry")
        file_path: Absolute path to the source file (None if inline module)
        ast: Parsed AST of the module
        exports: Set of symbols marked as public (`pub`)
        dependencies: Set of module names this module depends on
        is_inline: True if this is an inline `mod { }` block, not a file
    """

    name: str
    file_path: Optional[Path]
    ast: "Program"
    exports: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)
    is_inline: bool = False


@dataclass
class ModuleRegistry:
    """
    Cache of loaded modules.

    Tracks all modules that have been loaded during compilation,
    their load order for code generation, and modules currently
    being loaded (for cycle detection).

    Attributes:
        modules: Map from module name to ModuleInfo
        load_order: Order in which modules were fully loaded
        _loading: Set of modules currently being loaded (for cycle detection)
    """

    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    load_order: list[str] = field(default_factory=list)
    _loading: set[str] = field(default_factory=set)

    def get(self, name: str) -> Optional[ModuleInfo]:
        """Get a module by name, or None if not loaded."""
        return self.modules.get(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a module is already loaded."""
        return name in self.modules

    def is_loading(self, name: str) -> bool:
        """Check if a module is currently being loaded (cycle detection)."""
        return name in self._loading

    def start_loading(self, name: str) -> None:
        """Mark a module as currently being loaded."""
        self._loading.add(name)

    def finish_loading(self, name: str, module: ModuleInfo) -> None:
        """Mark a module as fully loaded and register it."""
        self._loading.discard(name)
        self.modules[name] = module
        self.load_order.append(name)

    def cancel_loading(self, name: str) -> None:
        """Cancel loading of a module (on error)."""
        self._loading.discard(name)

    def get_loading_chain(self) -> list[str]:
        """Get the current chain of modules being loaded."""
        return list(self._loading)


@dataclass
class DependencyGraph:
    """
    Graph of module dependencies for cycle detection and ordering.

    Each edge (A -> B) means module A depends on (imports) module B.

    Attributes:
        edges: Map from module name to set of modules it depends on
    """

    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_module(self, module: str) -> None:
        """Add a module to the graph if not already present."""
        if module not in self.edges:
            self.edges[module] = set()

    def add_dependency(self, module: str, depends_on: str) -> None:
        """
        Add a dependency edge: module depends on depends_on.

        Args:
            module: The module that has the dependency
            depends_on: The module being depended on
        """
        self.add_module(module)
        self.add_module(depends_on)
        self.edges[module].add(depends_on)

    def detect_cycle(self, start: str) -> Optional[list[str]]:
        """
        Detect if adding a dependency would create a cycle.

        Uses DFS to find a path from start back to itself.

        Args:
            start: The starting module to check

        Returns:
            The cycle path if one exists (e.g., ["a", "b", "c", "a"]),
            None if no cycle exists
        """
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> Optional[list[str]]:
            if node in path:
                # Found a cycle - return the cycle portion
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for dep in self.edges.get(node, set()):
                cycle = dfs(dep)
                if cycle:
                    return cycle

            path.pop()
            return None

        return dfs(start)

    def topological_sort(self) -> list[str]:
        """
        Return modules in topological order (dependencies first).

        Modules that don't depend on anything come first, then modules
        that only depend on those, etc. This ensures proper compilation
        order.

        Returns:
            List of module names in dependency order

        Raises:
            ValueError: If a cycle is detected
        """
        # Kahn's algorithm
        in_degree: dict[str, int] = {m: 0 for m in self.edges}

        # Calculate in-degrees
        for deps in self.edges.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Start with modules that have no dependencies on them
        queue: list[str] = [m for m, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            module = queue.pop(0)
            result.append(module)

            for dep in self.edges.get(module, set()):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)

        if len(result) != len(self.edges):
            # Cycle detected
            remaining = set(self.edges.keys()) - set(result)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        return result


# =============================================================================
# Known Python Modules
# =============================================================================

# Modules that should be treated as Python imports, not MathViz modules
PYTHON_MODULES = frozenset({
    # Standard library
    "math",
    "random",
    "os",
    "sys",
    "json",
    "re",
    "time",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    "pathlib",
    "io",
    "copy",
    "operator",
    "string",
    "textwrap",
    "struct",
    "decimal",
    "fractions",
    "statistics",
    "cmath",
    # Scientific/numeric
    "numpy",
    "np",
    "scipy",
    "pandas",
    "matplotlib",
    "sympy",
    "numba",
    # Manim
    "manim",
    "manimlib",
    # Other common libraries
    "PIL",
    "cv2",
    "sklearn",
    "torch",
    "tensorflow",
})


def is_python_module(module_path: tuple[str, ...]) -> bool:
    """
    Check if a module path refers to a Python (not MathViz) module.

    Args:
        module_path: Tuple of module path components (e.g., ("numpy", "linalg"))

    Returns:
        True if this is a known Python module
    """
    if not module_path:
        return False
    return module_path[0] in PYTHON_MODULES


# =============================================================================
# Module Loader
# =============================================================================


class ModuleLoader:
    """
    Resolves and loads MathViz modules.

    The loader handles:
    - Resolving `use` statements to actual files
    - Loading and parsing module files
    - Tracking dependencies between modules
    - Detecting circular dependencies
    - Collecting public exports from modules

    Resolution order for `use foo.bar`:
    1. ./foo/bar.mviz (relative to current file)
    2. ./foo/bar/mod.mviz (directory module)
    3. <root>/foo/bar.mviz
    4. <search_paths>/foo/bar.mviz

    Usage:
        loader = ModuleLoader(root_path=Path("."), search_paths=[Path("lib")])
        main_module = loader.load_file(Path("main.mviz"))

        for use_stmt in ast.statements:
            if isinstance(use_stmt, UseStatement):
                module = loader.resolve_use_statement(use_stmt, Path("main.mviz"))
    """

    def __init__(
        self,
        root_path: Path,
        search_paths: Optional[list[Path]] = None,
    ) -> None:
        """
        Initialize the module loader.

        Args:
            root_path: Root directory of the project
            search_paths: Additional directories to search for modules
        """
        self.root_path = root_path.resolve()
        self.search_paths = [p.resolve() for p in (search_paths or [])]
        self.registry = ModuleRegistry()
        self.dependency_graph = DependencyGraph()

    def load_file(self, file_path: Path) -> ModuleInfo:
        """
        Load a module from a file.

        Args:
            file_path: Path to the .mviz file

        Returns:
            ModuleInfo for the loaded module

        Raises:
            ModuleResolutionError: If the file cannot be found or loaded
            ModuleResolutionError: If a circular dependency is detected
        """
        from mathviz.compiler.lexer import Lexer
        from mathviz.compiler.parser import Parser
        from mathviz.utils.errors import ModuleResolutionError

        file_path = file_path.resolve()

        # Derive module name from file path
        module_name = self._path_to_module_name(file_path)

        # Check if already loaded
        if self.registry.is_loaded(module_name):
            return self.registry.get(module_name)  # type: ignore

        # Check for circular dependency
        if self.registry.is_loading(module_name):
            loading_chain = self.registry.get_loading_chain()
            cycle = loading_chain + [module_name]
            raise ModuleResolutionError(
                f"Circular dependency detected: {' -> '.join(cycle)}",
                module_path=module_name,
                cycle=cycle,
            )

        # Check file exists
        if not file_path.exists():
            raise ModuleResolutionError(
                f"Module file not found: {file_path}",
                module_path=module_name,
                search_paths=[self.root_path] + self.search_paths,
            )

        # Start loading
        self.registry.start_loading(module_name)
        self.dependency_graph.add_module(module_name)

        try:
            # Read and parse the file
            source = file_path.read_text(encoding="utf-8")
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()

            # Collect exports (pub items)
            exports = self._collect_exports(ast)

            # Collect dependencies from use statements
            dependencies: set[str] = set()
            for stmt in ast.statements:
                if hasattr(stmt, "__class__") and stmt.__class__.__name__ == "UseStatement":
                    dep_path = stmt.module_path
                    if not is_python_module(dep_path):
                        dep_name = ".".join(dep_path)
                        dependencies.add(dep_name)
                        self.dependency_graph.add_dependency(module_name, dep_name)

            # Create module info
            module = ModuleInfo(
                name=module_name,
                file_path=file_path,
                ast=ast,
                exports=exports,
                dependencies=dependencies,
                is_inline=False,
            )

            # Finish loading
            self.registry.finish_loading(module_name, module)

            return module

        except Exception as e:
            self.registry.cancel_loading(module_name)
            if isinstance(e, ModuleResolutionError):
                raise
            raise ModuleResolutionError(
                f"Failed to load module '{module_name}': {e}",
                module_path=module_name,
            ) from e

    def load_from_ast(
        self,
        ast: "Program",
        module_name: str,
        file_path: Optional[Path] = None,
    ) -> ModuleInfo:
        """
        Register a module from an already-parsed AST.

        This is useful for the main module which is already parsed.

        Args:
            ast: The parsed AST
            module_name: Name for this module
            file_path: Optional file path

        Returns:
            ModuleInfo for the module
        """
        if self.registry.is_loaded(module_name):
            return self.registry.get(module_name)  # type: ignore

        exports = self._collect_exports(ast)

        # Collect dependencies
        dependencies: set[str] = set()
        for stmt in ast.statements:
            if hasattr(stmt, "__class__") and stmt.__class__.__name__ == "UseStatement":
                dep_path = stmt.module_path
                if not is_python_module(dep_path):
                    dep_name = ".".join(dep_path)
                    dependencies.add(dep_name)

        module = ModuleInfo(
            name=module_name,
            file_path=file_path.resolve() if file_path else None,
            ast=ast,
            exports=exports,
            dependencies=dependencies,
            is_inline=False,
        )

        self.registry.modules[module_name] = module
        self.dependency_graph.add_module(module_name)

        return module

    def resolve_use_statement(
        self,
        use_stmt: "UseStatement",
        current_file: Path,
    ) -> Optional[ModuleInfo]:
        """
        Resolve a `use` statement to a module.

        Args:
            use_stmt: The UseStatement AST node
            current_file: Path to the file containing the use statement

        Returns:
            ModuleInfo for the resolved module, or None if it's a Python module

        Raises:
            ModuleResolutionError: If the module cannot be found
        """
        from mathviz.utils.errors import ModuleResolutionError

        module_path = use_stmt.module_path

        # Skip Python modules
        if is_python_module(module_path):
            return None

        # Try to resolve the module path to a file
        resolved_path = self._resolve_module_path(module_path, current_file.parent)

        if resolved_path is None:
            raise ModuleResolutionError(
                f"Module not found: {'.'.join(module_path)}",
                location=use_stmt.location,
                module_path=".".join(module_path),
                search_paths=[current_file.parent, self.root_path] + self.search_paths,
            )

        return self.load_file(resolved_path)

    def _resolve_module_path(
        self,
        module_path: tuple[str, ...],
        relative_to: Path,
    ) -> Optional[Path]:
        """
        Resolve a module path to an actual file.

        Search order:
        1. ./module/path.mviz (relative to current file)
        2. ./module/path/mod.mviz (directory module)
        3. <root>/module/path.mviz
        4. <search_paths>/module/path.mviz

        Args:
            module_path: Tuple of module path components
            relative_to: Directory to start relative search from

        Returns:
            Resolved Path if found, None otherwise
        """
        # Convert module path to file path segments
        path_segments = list(module_path)

        # Search locations in order
        search_dirs = [
            relative_to.resolve(),
            self.root_path,
            *self.search_paths,
        ]

        for search_dir in search_dirs:
            # Try direct file: ./module/path.mviz
            file_path = search_dir.joinpath(*path_segments).with_suffix(".mviz")
            if file_path.exists() and file_path.is_file():
                return file_path

            # Try directory module: ./module/path/mod.mviz
            dir_path = search_dir.joinpath(*path_segments, "mod.mviz")
            if dir_path.exists() and dir_path.is_file():
                return dir_path

        return None

    def _collect_exports(self, ast: "Program") -> set[str]:
        """
        Collect all public exports from a module AST.

        Currently, all top-level functions, classes, and module declarations
        are considered public exports. The `pub` keyword support is planned
        for future implementation.

        Note: The MathViz parser doesn't yet fully support tracking `pub`
        visibility on FunctionDef nodes. For now, we export all top-level
        items. Future versions will support explicit visibility control.

        Args:
            ast: The parsed module AST

        Returns:
            Set of exported symbol names
        """
        exports: set[str] = set()

        for stmt in ast.statements:
            stmt_class = stmt.__class__.__name__

            # Export all top-level functions
            if stmt_class == "FunctionDef":
                exports.add(stmt.name)

            # Export classes
            elif stmt_class == "ClassDef":
                exports.add(stmt.name)

            # Export scenes
            elif stmt_class == "SceneDef":
                exports.add(stmt.name)

            # Export module declarations
            elif stmt_class == "ModuleDecl":
                exports.add(stmt.name)

            # Export struct definitions
            elif stmt_class == "StructDef":
                exports.add(stmt.name)

            # Export enum definitions
            elif stmt_class == "EnumDef":
                exports.add(stmt.name)

            # Export trait definitions
            elif stmt_class == "TraitDef":
                exports.add(stmt.name)

            # Export const declarations
            elif stmt_class == "ConstDeclaration":
                exports.add(stmt.name)

            # Export let statements at module level (module constants)
            elif stmt_class == "LetStatement":
                if hasattr(stmt, "name"):
                    exports.add(stmt.name)

        return exports

    def _path_to_module_name(self, file_path: Path) -> str:
        """
        Convert a file path to a module name.

        Args:
            file_path: Absolute path to a .mviz file

        Returns:
            Module name (e.g., "lib.math" for "lib/math.mviz")
        """
        file_path = file_path.resolve()

        # Try to make path relative to root first
        rel_path: Optional[Path] = None
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            pass

        # If not relative to root, try search paths
        if rel_path is None:
            for search_path in self.search_paths:
                try:
                    rel_path = file_path.relative_to(search_path)
                    break
                except ValueError:
                    continue

        # If still no match, just use the file name
        if rel_path is None:
            rel_path = Path(file_path.stem)

        # Convert path to module name
        parts = list(rel_path.parts)

        # Remove .mviz extension from last part
        if parts and parts[-1].endswith(".mviz"):
            parts[-1] = parts[-1][:-5]

        # Handle mod.mviz -> parent name
        if parts and parts[-1] == "mod":
            parts = parts[:-1]

        return ".".join(parts) if parts else file_path.stem

    def check_circular_dependency(
        self,
        module: str,
        depends_on: str,
    ) -> Optional[list[str]]:
        """
        Check if adding a dependency would create a circular dependency.

        Args:
            module: The module that would have the dependency
            depends_on: The module being depended on

        Returns:
            The cycle path if one would be created, None otherwise
        """
        # Temporarily add the edge
        self.dependency_graph.add_dependency(module, depends_on)

        # Check for cycle
        cycle = self.dependency_graph.detect_cycle(module)

        # If cycle detected, we don't need to remove it (will raise error)
        # If no cycle, the edge stays (valid dependency)

        return cycle

    def check_visibility(
        self,
        module_name: str,
        symbol_name: str,
    ) -> bool:
        """
        Check if a symbol is accessible from outside its module.

        Args:
            module_name: Name of the module containing the symbol
            symbol_name: Name of the symbol to check

        Returns:
            True if the symbol is public, False if private
        """
        module = self.registry.get(module_name)
        if module is None:
            return False
        return symbol_name in module.exports

    def get_compilation_order(self) -> list[str]:
        """
        Get modules in compilation order (dependencies first).

        Returns:
            List of module names in the order they should be compiled
        """
        try:
            return self.dependency_graph.topological_sort()
        except ValueError:
            # Cycle detected - return modules in load order as fallback
            return self.registry.load_order


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ModuleInfo",
    "ModuleRegistry",
    "DependencyGraph",
    "ModuleLoader",
    "is_python_module",
    "PYTHON_MODULES",
]
