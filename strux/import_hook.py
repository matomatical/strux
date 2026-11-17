"""Apply recursive function checking to selected filesystem source imports.

Install before importing application modules::

    with strux.install_import_hook("my_package", exclude=("my_package.vendor",)):
        import my_package

The imported functions remain checked after the context exits. A persistent
installation can instead be removed with the returned handle's uninstall().
Already imported modules are unaffected; removal does not unwrap functions.

Only synchronous, non-generator function definitions are instrumented,
including methods and nested functions. Checking is placed inside existing
decorators, so JAX transformations trace checked functions. An explicit
@strux.checked overrides the hook's checking configuration at that boundary.
Lambda expressions, async functions, generators, class constructor methods
(__init__, __new__, __post_init__), and generated dataclass methods are left
alone. Functions called by constructors can still have checked boundaries.

Use @typing.no_type_check on a function or class to opt out. The spellings
@no_type_check and @typing.no_type_check also exclude nested definitions
syntactically. Jaxtyping's global disable configuration applies at call time.

The hook supports normal PathFinder/SourceFileLoader imports of Python source
files. Earlier custom import finders, zip imports, extensions, and bytecode-only
modules are left alone. Instrumented source is compiled without reading or
writing bytecode caches. Do not also instrument the same modules with another
checking framework's import hook.
"""

import ast
from collections.abc import Sequence
import functools
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
import sys

from strux.checking import _DEFAULT, checked


def install_import_hook(
    modules: str | Sequence[str],
    *,
    exclude: str | Sequence[str] = (),
    typechecker=_DEFAULT,
):
    """Install checked boundaries for named modules and their submodules.

    ``modules`` and ``exclude`` are dotted module names, not filesystem paths
    or arbitrary string prefixes. Exclusions take precedence within this hook.
    ``typechecker`` has the same meaning as in checked: the default is beartype
    (install strux[beartype]), and None establishes contexts without automatic
    checks. Returns a context manager with an idempotent uninstall() method.

    Installation is immediate. A context manager removes the hook on exit,
    including exceptional exits. It does not undo wrappers already installed.
    For overlapping strux hooks, the first installed matching hook wins.
    """
    modules = _module_names(modules)
    exclusions = _module_names(exclude)
    if not modules:
        raise ValueError("modules must contain at least one module name")
    if typechecker is not _DEFAULT and typechecker is not None:
        if not callable(typechecker):
            raise TypeError("typechecker must be a function decorator or None")
    finder = _Finder(modules, exclusions, typechecker)
    try:
        index = sys.meta_path.index(PathFinder)
    except ValueError:
        raise RuntimeError(
            "strux import hook requires PathFinder in sys.meta_path"
        ) from None
    sys.meta_path.insert(index, finder)
    return _ImportHook(finder)


def _module_names(names):
    if isinstance(names, str):
        names = (names,)
    if not isinstance(names, Sequence):
        raise TypeError("module names must be a string or a sequence of strings")
    for name in names:
        if not isinstance(name, str) or not all(
            part.isidentifier() for part in name.split(".")
        ):
            raise ValueError(f"expected a dotted module name, got {name!r}")
    return tuple(names)


def _matches(fullname, names):
    return any(fullname == name or fullname.startswith(name + ".") for name in names)


class _ImportHook:
    def __init__(self, finder):
        self._finder = finder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()

    def uninstall(self):
        """Stop instrumenting future imports; existing functions stay checked."""
        if self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)


class _Finder(MetaPathFinder):
    def __init__(self, modules, exclusions, typechecker):
        self.modules = modules
        self.exclusions = exclusions
        self.typechecker = typechecker

    def find_spec(self, fullname, path=None, target=None):
        if not _matches(fullname, self.modules) or _matches(fullname, self.exclusions):
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and type(spec.loader) is SourceFileLoader:
            spec.loader = _Loader(fullname, spec.loader.path, self.typechecker)
            return spec
        return None


class _Loader(SourceFileLoader):
    def __init__(self, fullname, path, typechecker):
        super().__init__(fullname, path)
        self.typechecker = typechecker

    def exec_module(self, module):
        source = self.get_source(self.name)
        # A single underscore avoids class-name mangling. Choosing a name absent
        # from the source keeps the helper separate from user bindings.
        helper = "_strux_import_checked"
        while helper in source:
            helper += "_"
        tree = ast.parse(source, filename=self.path)
        tree = _Transformer(helper).visit(tree)
        ast.fix_missing_locations(tree)
        module.__dict__[helper] = functools.partial(
            checked, typechecker=self.typechecker
        )
        code = compile(tree, self.path, "exec", dont_inherit=True)
        exec(code, module.__dict__)


def _no_type_check(node):
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "no_type_check":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "no_type_check":
            return True
    return False


class _YieldFinder(ast.NodeVisitor):
    """Find yields in one function body, excluding nested definitions."""

    found = False

    def visit_Yield(self, node):
        self.found = True

    visit_YieldFrom = visit_Yield

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


class _Transformer(ast.NodeTransformer):
    def __init__(self, helper):
        self.helper = helper
        self.in_class = False

    def visit_ClassDef(self, node):
        if _no_type_check(node):
            return node
        previous = self.in_class
        self.in_class = True
        self.generic_visit(node)
        self.in_class = previous
        return node

    def visit_FunctionDef(self, node):
        if _no_type_check(node):
            return node
        constructor = self.in_class and node.name in (
            "__init__",
            "__new__",
            "__post_init__",
        )
        yields = _YieldFinder()
        for statement in node.body:
            yields.visit(statement)
        if not constructor and not yields.found and isinstance(node, ast.FunctionDef):
            decorator = ast.Name(id=self.helper, ctx=ast.Load())
            node.decorator_list.append(ast.copy_location(decorator, node))
        previous = self.in_class
        self.in_class = False
        self.generic_visit(node)
        self.in_class = previous
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
