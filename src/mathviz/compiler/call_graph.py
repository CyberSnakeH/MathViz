"""
Call Graph Analysis for MathViz Compiler.

This module provides call graph construction and analysis capabilities for
determining function dependencies, detecting recursion, and establishing
compilation order.

Key features:
- Build call graphs from AST
- Detect direct and indirect recursion
- Find strongly connected components (mutually recursive functions)
- Compute topological sort for compilation order

Algorithms implemented:
- DFS-based cycle detection
- Tarjan's algorithm for strongly connected components
- Kahn's algorithm for topological sort
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from mathviz.compiler.ast_nodes import (
    BaseASTVisitor,
    Block,
    CallExpression,
    ClassDef,
    FunctionDef,
    Identifier,
    LambdaExpression,
    MemberAccess,
    ModuleDecl,
    Program,
    SceneDef,
)
from mathviz.utils.errors import SourceLocation


@dataclass(slots=True)
class CallSite:
    """
    Represents a single function call in the program.

    Attributes:
        caller: Name of the calling function
        callee: Name of the called function
        location: Source location of the call site
        is_recursive: Whether this call creates a recursive cycle
    """

    caller: str
    callee: str
    location: SourceLocation
    is_recursive: bool = False

    def __repr__(self) -> str:
        marker = " [recursive]" if self.is_recursive else ""
        return f"CallSite({self.caller} -> {self.callee}{marker})"


@dataclass(slots=True)
class CallGraphNode:
    """
    Represents a function in the call graph.

    Attributes:
        name: Function name
        calls: Set of function names this function calls
        called_by: Set of function names that call this function
        is_recursive: Whether this function calls itself directly
        in_cycle: Whether this function is part of a mutual recursion cycle
    """

    name: str
    calls: set[str] = field(default_factory=set)
    called_by: set[str] = field(default_factory=set)
    is_recursive: bool = False
    in_cycle: bool = False

    def __repr__(self) -> str:
        flags = []
        if self.is_recursive:
            flags.append("recursive")
        if self.in_cycle:
            flags.append("in_cycle")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"CallGraphNode({self.name}{flag_str})"


class CallGraphError(Exception):
    """Exception raised for call graph analysis errors."""

    def __init__(self, message: str, cycles: Optional[list[list[str]]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cycles = cycles or []


class CallGraph:
    """
    Represents the complete call graph of a program.

    The call graph tracks which functions call which other functions,
    supporting various analyses like cycle detection and topological ordering.
    """

    def __init__(self) -> None:
        """Initialize an empty call graph."""
        self.nodes: dict[str, CallGraphNode] = {}
        self.edges: list[CallSite] = []

    def add_function(self, name: str) -> None:
        """
        Add a function to the call graph.

        If the function already exists, this is a no-op.

        Args:
            name: The function name to add
        """
        if name not in self.nodes:
            self.nodes[name] = CallGraphNode(name=name)

    def add_call(
        self,
        caller: str,
        callee: str,
        location: SourceLocation,
    ) -> None:
        """
        Record a function call from caller to callee.

        Both functions are added to the graph if they don't exist.
        Direct self-recursion is detected and marked immediately.

        Args:
            caller: Name of the calling function
            callee: Name of the called function
            location: Source location of the call
        """
        # Ensure both functions exist in the graph
        self.add_function(caller)
        self.add_function(callee)

        # Check for direct self-recursion
        is_recursive = caller == callee
        if is_recursive:
            self.nodes[caller].is_recursive = True
            self.nodes[caller].in_cycle = True

        # Record the edge
        call_site = CallSite(
            caller=caller,
            callee=callee,
            location=location,
            is_recursive=is_recursive,
        )
        self.edges.append(call_site)

        # Update adjacency lists
        self.nodes[caller].calls.add(callee)
        self.nodes[callee].called_by.add(caller)

    def get_callers(self, func: str) -> set[str]:
        """
        Get all functions that call the given function.

        Args:
            func: Function name to query

        Returns:
            Set of function names that call func, empty set if func not found
        """
        if func not in self.nodes:
            return set()
        return self.nodes[func].called_by.copy()

    def get_callees(self, func: str) -> set[str]:
        """
        Get all functions called by the given function.

        Args:
            func: Function name to query

        Returns:
            Set of function names called by func, empty set if func not found
        """
        if func not in self.nodes:
            return set()
        return self.nodes[func].calls.copy()

    def find_cycles(self) -> list[list[str]]:
        """
        Find all cycles in the call graph using DFS.

        This detects both direct recursion (f calls f) and indirect
        recursion (f calls g, g calls f).

        Returns:
            List of cycles, where each cycle is a list of function names
            forming the cycle path. For example, [["f", "g", "f"]] means
            f calls g and g calls f.
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.nodes[node].calls:
                if neighbor not in self.nodes:
                    continue

                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle - extract it from the path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        # Run DFS from each unvisited node
        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def topological_sort(self) -> list[str]:
        """
        Compute a topological ordering of functions using Kahn's algorithm.

        Functions with no dependencies come first, allowing them to be
        compiled before functions that call them. This is useful for
        determining compilation order in a single-pass compiler.

        Returns:
            List of function names in topological order (callees before callers)

        Raises:
            CallGraphError: If the graph contains cycles, topological sort
                           is not possible
        """
        # Compute in-degrees (number of functions this function calls
        # that need to be compiled first)
        # For compilation order, we reverse the edges:
        # A function must be compiled after all functions it calls
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}

        for node in self.nodes.values():
            for callee in node.calls:
                if callee in in_degree:
                    in_degree[node.name] += 1

        # Start with nodes that don't call any other functions in the graph
        queue: deque[str] = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)

        result: list[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Decrement in-degree for callers of current function
            for caller in self.nodes[current].called_by:
                if caller in in_degree:
                    in_degree[caller] -= 1
                    if in_degree[caller] == 0:
                        queue.append(caller)

        # Check if we processed all nodes
        if len(result) != len(self.nodes):
            cycles = self.find_cycles()
            cycle_desc = ", ".join(" -> ".join(cycle) for cycle in cycles)
            raise CallGraphError(
                f"Cannot compute topological sort: graph contains cycles ({cycle_desc})",
                cycles=cycles,
            )

        return result

    def get_strongly_connected_components(self) -> list[set[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.

        A strongly connected component (SCC) is a maximal set of nodes where
        every node is reachable from every other node. In the context of
        call graphs, SCCs represent sets of mutually recursive functions.

        This implementation uses Tarjan's algorithm with a single DFS pass.

        Returns:
            List of SCCs, where each SCC is a set of function names.
            SCCs are returned in reverse topological order (callees first).
        """
        # Tarjan's algorithm state
        index_counter = [0]  # Mutable counter in closure
        stack: list[str] = []
        lowlinks: dict[str, int] = {}
        index: dict[str, int] = {}
        on_stack: set[str] = set()
        sccs: list[set[str]] = []

        def strongconnect(node: str) -> None:
            """Tarjan's DFS function."""
            # Set the depth index for this node
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            # Consider successors
            for successor in self.nodes[node].calls:
                if successor not in self.nodes:
                    continue

                if successor not in index:
                    # Successor has not been visited; recurse
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    # Successor is on stack, so part of current SCC
                    lowlinks[node] = min(lowlinks[node], index[successor])

            # If node is a root of an SCC, pop the SCC from stack
            if lowlinks[node] == index[node]:
                scc: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)

        # Run Tarjan's algorithm from each unvisited node
        for node in self.nodes:
            if node not in index:
                strongconnect(node)

        return sccs

    def mark_cycles(self) -> None:
        """
        Analyze the graph and mark all nodes that are part of cycles.

        This sets the `in_cycle` flag on all CallGraphNode instances
        that are part of any cycle (including single-node self-recursion).
        Also marks the corresponding CallSite edges as recursive.
        """
        # Find all SCCs with more than one node (mutual recursion)
        sccs = self.get_strongly_connected_components()

        for scc in sccs:
            if len(scc) > 1:
                # All functions in this SCC are mutually recursive
                for func_name in scc:
                    self.nodes[func_name].in_cycle = True

        # Direct recursion is already marked in add_call
        # Mark edges that participate in cycles
        cycles = self.find_cycles()
        cycle_edges: set[tuple[str, str]] = set()

        for cycle in cycles:
            for i in range(len(cycle) - 1):
                cycle_edges.add((cycle[i], cycle[i + 1]))

        for edge in self.edges:
            if (edge.caller, edge.callee) in cycle_edges:
                edge.is_recursive = True

    def get_roots(self) -> set[str]:
        """
        Get functions that are not called by any other function.

        These are typically entry points (main functions, scene constructors, etc.)

        Returns:
            Set of function names with no callers
        """
        return {name for name, node in self.nodes.items() if not node.called_by}

    def get_leaves(self) -> set[str]:
        """
        Get functions that do not call any other function.

        These are leaf functions in the call tree.

        Returns:
            Set of function names that make no calls
        """
        return {name for name, node in self.nodes.items() if not node.calls}

    def get_transitive_callees(self, func: str) -> set[str]:
        """
        Get all functions reachable from the given function.

        This includes direct callees and their callees, recursively.

        Args:
            func: Starting function name

        Returns:
            Set of all function names reachable from func
        """
        if func not in self.nodes:
            return set()

        visited: set[str] = set()
        queue: deque[str] = deque([func])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for callee in self.nodes[current].calls:
                if callee in self.nodes and callee not in visited:
                    queue.append(callee)

        visited.discard(func)  # Don't include the starting function
        return visited

    def get_transitive_callers(self, func: str) -> set[str]:
        """
        Get all functions that can reach the given function.

        This includes direct callers and their callers, recursively.

        Args:
            func: Target function name

        Returns:
            Set of all function names that can reach func
        """
        if func not in self.nodes:
            return set()

        visited: set[str] = set()
        queue: deque[str] = deque([func])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for caller in self.nodes[current].called_by:
                if caller in self.nodes and caller not in visited:
                    queue.append(caller)

        visited.discard(func)  # Don't include the starting function
        return visited

    def __repr__(self) -> str:
        return f"CallGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __str__(self) -> str:
        lines = ["CallGraph:"]
        for node in sorted(self.nodes.values(), key=lambda n: n.name):
            lines.append(f"  {node.name}:")
            if node.calls:
                lines.append(f"    calls: {sorted(node.calls)}")
            if node.called_by:
                lines.append(f"    called_by: {sorted(node.called_by)}")
            if node.is_recursive:
                lines.append("    [self-recursive]")
            if node.in_cycle and not node.is_recursive:
                lines.append("    [in mutual recursion]")
        return "\n".join(lines)


class CallGraphBuilder(BaseASTVisitor):
    """
    AST visitor that builds a call graph from a program.

    This visitor traverses the AST and records:
    - All function definitions
    - All call expressions within function bodies

    Usage:
        builder = CallGraphBuilder()
        call_graph = builder.build(program)
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._graph: CallGraph = CallGraph()
        self._current_function: Optional[str] = None
        self._function_scope_stack: list[str] = []
        self._defined_functions: set[str] = set()

    def build(self, program: Program) -> CallGraph:
        """
        Build the call graph for the program.

        Args:
            program: The parsed program AST

        Returns:
            The constructed call graph with all functions and calls
        """
        self._graph = CallGraph()
        self._current_function = None
        self._function_scope_stack = []
        self._defined_functions = set()

        # First pass: collect all function definitions
        self._collect_definitions(program)

        # Second pass: analyze function bodies for calls
        self.visit(program)

        # Mark cycles and mutual recursion
        self._graph.mark_cycles()

        return self._graph

    def _collect_definitions(self, program: Program) -> None:
        """First pass: collect all function names."""
        collector = _FunctionCollector()
        collector.visit(program)
        self._defined_functions = collector.functions

        for func_name in self._defined_functions:
            self._graph.add_function(func_name)

    def visit_function_def(self, node: FunctionDef) -> None:
        """Visit a function definition and analyze its body for calls."""
        func_name = node.name

        # Push this function onto the scope stack
        self._function_scope_stack.append(func_name)
        self._current_function = func_name

        # Visit the function body to find calls
        if node.body:
            self.visit(node.body)

        # Pop the function from the scope stack
        self._function_scope_stack.pop()
        self._current_function = (
            self._function_scope_stack[-1] if self._function_scope_stack else None
        )

    def visit_lambda_expression(self, node: LambdaExpression) -> None:
        """
        Visit a lambda expression.

        Lambdas are anonymous, so we attribute their calls to the
        enclosing function.
        """
        self.visit(node.body)

    def visit_class_def(self, node: ClassDef) -> None:
        """Visit a class definition and process its methods."""
        if node.body:
            # Methods within a class are prefixed with class name
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    method_name = f"{node.name}.{stmt.name}"
                    self._graph.add_function(method_name)
                    self._defined_functions.add(method_name)

                    self._function_scope_stack.append(method_name)
                    self._current_function = method_name

                    if stmt.body:
                        self.visit(stmt.body)

                    self._function_scope_stack.pop()
                    self._current_function = (
                        self._function_scope_stack[-1] if self._function_scope_stack else None
                    )
                else:
                    self.visit(stmt)

    def visit_scene_def(self, node: SceneDef) -> None:
        """Visit a Manim scene definition and process its methods."""
        if node.body:
            # Scene methods are prefixed with scene name
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    method_name = f"{node.name}.{stmt.name}"
                    self._graph.add_function(method_name)
                    self._defined_functions.add(method_name)

                    self._function_scope_stack.append(method_name)
                    self._current_function = method_name

                    if stmt.body:
                        self.visit(stmt.body)

                    self._function_scope_stack.pop()
                    self._current_function = (
                        self._function_scope_stack[-1] if self._function_scope_stack else None
                    )
                else:
                    self.visit(stmt)

    def visit_module_decl(self, node: ModuleDecl) -> None:
        """Visit a module declaration and process its functions."""
        # Functions within a module are prefixed with module name
        for stmt in node.body.statements:
            if isinstance(stmt, FunctionDef):
                func_name = f"{node.name}.{stmt.name}"
                self._graph.add_function(func_name)
                self._defined_functions.add(func_name)

                self._function_scope_stack.append(func_name)
                self._current_function = func_name

                if stmt.body:
                    self.visit(stmt.body)

                self._function_scope_stack.pop()
                self._current_function = (
                    self._function_scope_stack[-1] if self._function_scope_stack else None
                )
            else:
                self.visit(stmt)

    def visit_call_expression(self, node: CallExpression) -> None:
        """Visit a call expression and record the call edge."""
        if self._current_function is not None:
            callee_name = self._extract_callee_name(node)

            if callee_name and callee_name in self._defined_functions:
                location = node.location or SourceLocation(line=0, column=0)
                self._graph.add_call(
                    caller=self._current_function,
                    callee=callee_name,
                    location=location,
                )

        # Continue traversing into the callee and arguments
        self.visit(node.callee)
        for arg in node.arguments:
            self.visit(arg)

    def _extract_callee_name(self, node: CallExpression) -> Optional[str]:
        """
        Extract the function name from a call expression.

        Handles:
        - Simple calls: foo()
        - Member calls: obj.method() (for self.method, Module.func, etc.)

        Args:
            node: The call expression node

        Returns:
            The function name if it can be determined, None otherwise
        """
        callee = node.callee

        if isinstance(callee, Identifier):
            return callee.name

        if isinstance(callee, MemberAccess):
            # Handle method calls like self.method or Module.function
            # For now, only handle simple cases
            if isinstance(callee.object, Identifier):
                obj_name = callee.object.name
                method_name = callee.member

                # Check if it's a known qualified name
                qualified_name = f"{obj_name}.{method_name}"
                if qualified_name in self._defined_functions:
                    return qualified_name

                # For 'self' calls, try to find the method in the current class
                if obj_name == "self" and self._current_function:
                    # Extract class name from current function
                    if "." in self._current_function:
                        class_name = self._current_function.split(".")[0]
                        class_method = f"{class_name}.{method_name}"
                        if class_method in self._defined_functions:
                            return class_method

        return None


class _FunctionCollector(BaseASTVisitor):
    """Helper visitor to collect all function names in a program."""

    def __init__(self) -> None:
        self.functions: set[str] = set()

    def visit_function_def(self, node: FunctionDef) -> None:
        self.functions.add(node.name)
        # Don't recurse into nested functions here

    def visit_class_def(self, node: ClassDef) -> None:
        if node.body:
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    self.functions.add(f"{node.name}.{stmt.name}")

    def visit_scene_def(self, node: SceneDef) -> None:
        if node.body:
            for stmt in node.body.statements:
                if isinstance(stmt, FunctionDef):
                    self.functions.add(f"{node.name}.{stmt.name}")

    def visit_module_decl(self, node: ModuleDecl) -> None:
        for stmt in node.body.statements:
            if isinstance(stmt, FunctionDef):
                self.functions.add(f"{node.name}.{stmt.name}")
