"""Imported boundaries have the same scopes and contracts as checked functions."""

import contextlib
import importlib
import inspect
from pathlib import Path
import py_compile
import sys
import textwrap
import uuid

import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Float, TypeCheckError
import pytest

import strux


@strux.struct
class HookArrayItem:
    values: Float[Array, "2"]


@pytest.fixture
def source_modules(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    prefix = "strux_hook_cases_" + uuid.uuid4().hex
    package = tmp_path / prefix
    package.mkdir()
    (package / "__init__.py").write_text("")

    def write(name, source):
        path = tmp_path.joinpath(*name.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
        importlib.invalidate_caches()
        return path

    yield prefix, write
    for name in tuple(sys.modules):
        if name.startswith(prefix):
            del sys.modules[name]


@pytest.fixture(params=["explicit", "hook"])
def checked_module(request, source_modules):
    prefix, write = source_modules

    def load(source):
        source = source.replace(
            "@CHECK", "@strux.checked" if request.param == "explicit" else ""
        )
        name = prefix + ".example"
        write(name, source)
        context = (
            strux.install_import_hook(prefix)
            if request.param == "hook"
            else contextlib.nullcontext()
        )
        with context:
            return importlib.import_module(name)

    return load


def test_argument_body_return_share_scope_and_nested_calls_are_independent(
    checked_module,
):
    module = checked_module("""
        import strux
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        @CHECK
        def body_mismatch(x: Float[Array, "n"]):
            assert isinstance(jnp.ones(3), Float[Array, "n"])

        @CHECK
        def body_return(x) -> Float[Array, "n"]:
            assert isinstance(x, Float[Array, "n"])
            return jnp.ones(3)

        @CHECK
        def inner(x: Float[Array, "n"]) -> Float[Array, "n"]:
            assert isinstance(x, Float[Array, "n"])
            return x

        @CHECK
        def outer(x: Float[Array, "n"]) -> Float[Array, "n"]:
            inner(jnp.ones(3))
            try:
                body_mismatch(x)
            except AssertionError:
                pass
            assert isinstance(x, Float[Array, "n"])
            return x
    """)
    value = jnp.ones(2)
    with pytest.raises(AssertionError):
        module.body_mismatch(value)
    with pytest.raises(TypeCheckError, match="return value.*body_return"):
        module.body_return(value)
    assert module.outer(value) is value
    assert module.outer(jnp.ones(4)).shape == (4,)


def test_recursive_fields_returns_and_shared_batch_dimensions(checked_module):
    module = checked_module("""
        import strux
        from jaxtyping import Array, Float
        from test_import_hook import HookArrayItem as Item

        @strux.struct
        class Holder:
            item: Item

        @CHECK
        def identity(value: Holder) -> Holder:
            return value

        @CHECK
        def wrong(value: Item) -> Item:
            return value.replace(values="wrong")

        @CHECK
        def batch(value: strux.Batch[Item, "n"], x: Float[Array, "n"]):
            return value

        @CHECK
        def shortened(value: strux.Batch[Item, "n"]) -> strux.Batch[Item, "n"]:
            return value[:1]
    """)
    item = module.Item(jnp.ones(2))
    holder = module.Holder(item)
    assert module.identity(holder) is holder
    malformed = item.replace(values="wrong")
    with pytest.raises(TypeCheckError, match="parameters.*identity"):
        module.identity(module.Holder(malformed))
    with pytest.raises(TypeCheckError, match="return value.*wrong"):
        module.wrong(item)
    batch = module.Item(jnp.ones((3, 2)))
    assert module.batch(batch, jnp.ones(3)) is batch
    with pytest.raises(TypeCheckError, match="n=3"):
        module.batch(batch, jnp.ones(4))
    with pytest.raises(TypeCheckError, match="return value.*shortened"):
        module.shortened(batch)


def test_local_references_self_and_descriptors(checked_module):
    module = checked_module("""
        from __future__ import annotations
        from typing import Self
        import strux

        @CHECK
        def make_class():
            @strux.struct
            class Local:
                value: int

                @CHECK
                def keep(self: Self, other: Local) -> Self:
                    return other

                @classmethod
                @CHECK
                def make(cls: type[Self], value: int) -> Self:
                    return cls(value)

                @staticmethod
                @CHECK
                def identity(value: Local) -> Local:
                    return value

                @property
                @CHECK
                def copy(self: Self) -> Self:
                    return self

            class Child(Local):
                pass

            @CHECK
            def identity(value: Local) -> Local:
                return value
            return Local, Child, identity
    """)
    local, child_class, identity = module.make_class()
    value = local.make(1)
    assert identity(value) is value
    assert value.keep(value) is value
    assert local.identity(value) is value
    assert value.copy is value
    malformed = local("wrong")
    for function in (identity, local.identity, value.keep):
        with pytest.raises(TypeCheckError):
            function(malformed)
    with pytest.raises(TypeCheckError):
        _ = malformed.copy

    child = child_class.make(1)
    assert type(child) is child_class
    with pytest.raises(TypeCheckError, match="return value"):
        child.keep(value)


def test_jax_decorator_order_transforms_and_disabled_results(
    checked_module, monkeypatch
):
    module = checked_module("""
        import jax
        import strux
        from jaxtyping import Array, Float
        from test_import_hook import HookArrayItem as Item

        @jax.jit
        @CHECK
        def step(item: Item) -> Item:
            return item.replace(values=item.values + 1)

        @CHECK
        def scan_step(item: Item, increment: Float[Array, ""]) -> tuple[Item, Item]:
            updated = item.replace(values=item.values + increment)
            return updated, updated

        @jax.custom_jvp
        @CHECK
        def square(x: Float[Array, ""]) -> Float[Array, ""]:
            return x * x

        @square.defjvp
        def square_jvp(primals, tangents):
            (x,), (dx,) = primals, tangents
            return square(x), 2 * x * dx
    """)
    item = module.Item(jnp.ones(2))
    assert jnp.array_equal(module.step(item).values, jnp.full(2, 2.0))
    with pytest.raises(TypeCheckError):
        module.step(module.Item(jnp.ones(3)))
    batch = module.Item(jnp.ones((3, 2)))
    assert jax.vmap(module.step)(batch).values.shape == (3, 2)
    final, history = jax.jit(lambda x: jax.lax.scan(module.scan_step, x, jnp.ones(3)))(
        item
    )
    assert jnp.array_equal(final.values, jnp.full(2, 4.0))
    assert history.values.shape == (3, 2)
    assert jax.grad(module.square)(jnp.array(2.0)) == 4.0
    enabled = module.scan_step(item, jnp.array(1.0))[0]
    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)
    disabled = module.scan_step(item, jnp.array(1.0))[0]
    assert jnp.array_equal(enabled.values, disabled.values)


def test_method_annotation_resolution_retains_class_type_parameter_scope(
    checked_module,
):
    module = checked_module("""
        from __future__ import annotations
        import strux

        class Handler[T: int]:
            @CHECK
            def identity(self, value: T) -> T:
                return value

            @CHECK
            def own_parameter[T: str](self, value: T) -> T:
                return value

        class Child(Handler[int]):
            pass
    """)
    for cls in (module.Handler, module.Child):
        instance = cls()
        assert instance.identity(1) == 1
        assert instance.own_parameter("value") == "value"
        with pytest.raises(TypeCheckError):
            instance.identity("wrong")
        with pytest.raises(TypeCheckError):
            instance.own_parameter(1)


def test_hook_skips_constructors_unsupported_functions_and_opt_outs(source_modules):
    prefix, write = source_modules
    name = prefix + ".skips"
    write(
        name,
        """
        from dataclasses import dataclass
        from typing import no_type_check
        import strux

        @strux.struct
        class Item:
            value: int
            def __post_init__(self: "Item"):
                pass

        class Custom:
            def __new__(cls, value: int):
                return super().__new__(cls)
            def __init__(self, value: int):
                self.value = value

        @dataclass
        class Ordinary:
            value: int

        async def asynchronous(value: int) -> int:
            return value

        def generator(value: int):
            yield value

        async def async_generator(value: int):
            yield value

        def factory(value: int):
            def nested_generator():
                yield value
            return nested_generator()

        @no_type_check
        def excluded(value: int):
            def nested(value: int):
                return value
            return nested(value)

        @no_type_check
        class Excluded:
            def method(self, value: int):
                return value
    """,
    )
    with strux.install_import_hook(prefix):
        module = importlib.import_module(name)
    assert module.Item("wrong").replace(value="also wrong").value == "also wrong"
    assert module.Custom("wrong").value == "wrong"
    assert module.Ordinary("wrong").value == "wrong"
    assert inspect.iscoroutinefunction(module.asynchronous)
    assert inspect.isasyncgenfunction(module.async_generator)
    assert inspect.isgeneratorfunction(module.generator)
    assert list(module.generator("wrong")) == ["wrong"]
    assert list(module.factory(1)) == [1]
    with pytest.raises(TypeCheckError):
        module.factory("wrong")
    assert module.excluded("wrong") == "wrong"
    assert module.Excluded().method("wrong") == "wrong"


def test_hook_scope_removal_existing_imports_and_bytecode(source_modules):
    prefix, write = source_modules
    source = '''
        """Preserved module docstring."""
        from __future__ import annotations
        _strux_import_checked = "user binding"
        def identity(value: int) -> int:
            return value
    '''
    names = [
        prefix + suffix
        for suffix in (".before", ".during", ".excluded", ".after", "_other")
    ]
    paths = {name: write(name, source) for name in names}
    before = importlib.import_module(names[0])
    bytecode = Path(py_compile.compile(str(paths[names[1]]), doraise=True))
    cached = bytecode.read_bytes()
    with pytest.raises(RuntimeError, match="exit"):
        with strux.install_import_hook(prefix, exclude=names[2]) as hook:
            during, excluded, unrelated = (
                importlib.import_module(name) for name in (names[1], names[2], names[4])
            )
            raise RuntimeError("exit")
    hook.uninstall()
    after = importlib.import_module(names[3])
    assert bytecode.read_bytes() == cached
    assert during.__doc__ == "Preserved module docstring."
    assert during._strux_import_checked == "user binding"
    for module in (before, excluded, unrelated, after):
        assert module.identity("wrong") == "wrong"
    with pytest.raises(TypeCheckError):
        during.identity("wrong")
    # Ordinary imports must not reuse instrumented bytecode after removal.
    del sys.modules[names[1]]
    assert importlib.import_module(names[1]).identity("wrong") == "wrong"


def test_explicit_checker_overrides_hook_without_duplicate_contexts(
    source_modules, monkeypatch
):
    prefix, write = source_modules
    name = prefix + ".explicit"
    write(
        name,
        """
        import strux
        from strux import checked as custom_checked
        from jaxtyping import Array, Float

        @custom_checked(typechecker=None)
        def unchecked(value: int):
            return value

        @custom_checked
        def body(value: Float[Array, "n"]):
            return value

        def unresolved(value: "Missing"):
            return value
    """,
    )
    with strux.install_import_hook(prefix):
        module = importlib.import_module(name)
    assert module.unchecked("wrong") == "wrong"
    assert not hasattr(module.body.__wrapped__, "__wrapped__")
    with pytest.raises(TypeCheckError):
        module.body("wrong")
    with pytest.raises(TypeError, match="could not resolve annotations"):
        module.unresolved(1)
    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)
    assert module.unresolved(1) == 1


def test_hook_context_only_backend(source_modules):
    prefix, write = source_modules
    name = prefix + ".context_only"
    write(
        name,
        """
        import jax.numpy as jnp
        from jaxtyping import Array, Float
        def identity(value: str):
            assert isinstance(jnp.ones(value), Float[Array, "n"])
            return value
        def mismatch():
            assert isinstance(jnp.ones(2), Float[Array, "n"])
            assert isinstance(jnp.ones(3), Float[Array, "n"])
    """,
    )
    with strux.install_import_hook(prefix, typechecker=None):
        module = importlib.import_module(name)
    assert module.identity(2) == 2
    assert module.identity(3) == 3
    with pytest.raises(AssertionError):
        module.mismatch()


@pytest.mark.parametrize("modules", [[], "", "pkg/child", ["pkg", 42]])
def test_invalid_module_selection(modules):
    with pytest.raises(ValueError):
        strux.install_import_hook(modules)


def test_invalid_backend():
    with pytest.raises(TypeError, match="typechecker"):
        strux.install_import_hook("example", typechecker="strux.checked")
