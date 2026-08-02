"""
Tests for schema compilation, including annotation-resolution regressions
(future-style string annotations, forward references).
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int

import strux

from example_structs import Environment


# # #
# Schema compilation


class TestSchema:
    def test_schema_is_cached_on_class(self):
        assert strux.schema(Environment) is strux.schema(Environment)

    def test_schema_repr_lists_fields(self):
        rendered = str(strux.schema(Environment))
        assert "hero_pos" in rendered
        assert "walls" in rendered

    def test_subclass_gets_own_schema(self):
        @strux.struct
        class Base:
            steps: Int[Array, ""]

        @strux.struct
        class Derived(Base):
            walls: Bool[Array, "h w"]

        assert set(strux.schema(Base).fields) == {"steps"}
        assert set(strux.schema(Derived).fields) == {"steps", "walls"}

    def test_bad_annotations_raise_lazily(self):
        # decoration succeeds; the schema (and hence the error) is built
        # at first construction
        from typing import Any

        @strux.struct
        class Bad:
            x: Any

        with pytest.raises(strux.SchemaError, match="array-leaved pytrees"):
            Bad(x=jnp.zeros(3))

    def test_object_and_callable_rejected(self):
        from typing import Callable

        @strux.struct
        class BadObject:
            x: object

        with pytest.raises(strux.SchemaError):
            BadObject(x=jnp.zeros(3))

        @strux.struct
        class BadCallable:
            f: Callable

        with pytest.raises(strux.SchemaError, match="mark the field static"):
            BadCallable(f=lambda x: x)

    def test_static_fields_may_be_anything(self):
        from typing import Any, Callable

        @strux.struct(static_fieldnames=("f", "meta"))
        class Fine:
            x: Float[Array, ""]
            f: Callable
            meta: Any

        Fine(x=jnp.float32(1.0), f=lambda v: v, meta=object())

    def test_non_leading_variadic_rejected(self):
        @strux.struct
        class Bad:
            x: Float[Array, "n ..."]

        with pytest.raises(strux.SchemaError, match="variadic"):
            Bad(x=jnp.zeros((2, 3)))

    def test_unresolvable_forward_reference_raises_clearly(self):
        @strux.struct
        class Bad:
            x: "NotDefinedAnywhere"  # noqa: F821

        with pytest.raises(strux.SchemaError, match="cannot resolve"):
            Bad(x=jnp.zeros(3))

    def test_foreign_registered_dataclass(self):
        # a plain jax.tree_util.register_dataclass dataclass (the hijax
        # style) gets a schema via the same parser, with metadata-static
        # fields excluded
        import dataclasses as dc
        import functools

        @functools.partial(
            jax.tree_util.register_dataclass,
            data_fields=("pos",),
            meta_fields=("name",),
        )
        @dc.dataclass(frozen=True)
        class Foreign:
            pos: jax.Array
            name: str = dc.field(metadata=dict(static=True))

        sch = strux.schema(Foreign)
        assert set(sch.fields) == {"pos"}


# # #
# Annotation-resolution regressions (future-style strings, forward refs)


class TestAnnotationResolution:
    def test_future_style_string_annotations(self):
        # equivalent to `from __future__ import annotations`: dataclasses
        # store the strings; the schema resolves them lazily in the
        # declaring module's namespace
        @strux.struct
        class Stringy:
            x: "Float[Array, ''] | None"
            count: "Int[Array, '']"

        s = Stringy(x=jnp.zeros(()), count=jnp.int32(0))
        assert s.shape == ()
        batched = Stringy(x=jnp.zeros(4), count=jnp.zeros(4, jnp.int32))
        assert batched.shape == (4,)
        assert isinstance(batched, Stringy["batch"])

    def test_self_referential_struct(self):
        # the class name is not bound at decoration time, but the schema
        # is built lazily, and the class itself is in scope by then
        @strux.struct
        class Linked:
            value: Float[Array, ""]
            prev: "Linked | None"

        first = Linked(value=jnp.float32(0.0), prev=None)
        second = Linked(value=jnp.float32(1.0), prev=first)
        assert second.shape == ()

    def test_inherited_field_resolves_in_declaring_module(self):
        # a subclass's inherited fields resolve against the class that
        # declared them (exercised via the shared test structs)
        @strux.struct
        class Extended(Environment):
            score: Float[Array, ""]

        e = Extended(
            hero_pos=jnp.zeros(2, jnp.int32),
            goal_pos=jnp.zeros(2, jnp.int32),
            walls=jnp.zeros((5, 5), bool),
            score=jnp.float32(0.0),
        )
        assert e.shape == ()
