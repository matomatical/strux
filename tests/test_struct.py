"""
Tests for the @struct decorator: field name collision guards and
__str__/__format__ method resolution.
"""

import warnings

import pytest

import strux


# # #
# Field name collision guards


class TestFieldCollisions:
    def test_size_field_warns(self):
        with pytest.warns(UserWarning, match="field named 'size'"):
            @strux.struct
            class HasSize:
                size: int
                x: float

    def test_size_field_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            @strux.struct
            class HasSize:
                size: int
                x: float
        obj = HasSize(size=3, x=1.0)
        assert obj.size == 3

    def test_replace_field_warns(self):
        with pytest.warns(UserWarning, match="field named 'replace'"):
            @strux.struct
            class HasReplace:
                replace: int
                x: float

    def test_replace_field_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            @strux.struct
            class HasReplace:
                replace: int
                x: float
        obj = HasReplace(replace=42, x=1.0)
        assert obj.replace == 42

    def test_no_warning_without_collision(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            @strux.struct
            class Normal:
                x: float
                y: float


# # #
# Format/str/repr method resolution


class TestMethodResolution:
    """
    Test that strux's auto-assigned __format__ delegates to str(self) for
    empty format specs, and uses tree_format parsing for non-empty specs.
    User overrides always win.
    """

    # -- __str__ / __format__ interaction (the interesting part) --

    def test_neither_overridden(self):
        @strux.struct
        class S:
            x: int
        obj = S(x=1)
        expected = strux.to_str(obj)
        assert str(obj) == expected
        assert f'{obj}' == expected
        assert f'{obj:0}' == strux.to_str(obj, max_depth=0)

    def test_str_overridden(self):
        @strux.struct
        class S:
            x: int
            def __str__(self):
                return "custom_str"
        obj = S(x=1)
        assert str(obj) == "custom_str"
        assert f'{obj}' == "custom_str"
        # non-empty spec still uses strux tree_format
        assert f'{obj:0}' == strux.to_str(obj, max_depth=0)

    def test_format_overridden(self):
        @strux.struct
        class S:
            x: int
            def __format__(self, spec):
                return f"custom_format:{spec}"
        obj = S(x=1)
        assert str(obj) == strux.to_str(obj)
        assert f'{obj}' == "custom_format:"
        assert f'{obj:2}' == "custom_format:2"

    def test_str_and_format_overridden(self):
        @strux.struct
        class S:
            x: int
            def __str__(self):
                return "custom_str"
            def __format__(self, spec):
                return "custom_format"
        obj = S(x=1)
        assert str(obj) == "custom_str"
        assert f'{obj}' == "custom_format"


# # #
# Batch dunders: user definitions are preserved with a warning


class TestBatchDunderResolution:
    def test_user_getitem_preserved_with_warning(self):
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        with pytest.warns(UserWarning, match="defines __getitem__"):
            @strux.struct
            class Lookup:
                table: Float[Array, "n"]
                def __getitem__(self, index):
                    return "custom"

        obj = Lookup(table=jnp.ones(3))
        assert obj[0] == "custom"
        # the module-level function remains available
        indexed = strux.tree_getitem(
            Lookup(table=jnp.ones((4, 3))), 0,
        )
        assert indexed.table.shape == (3,)


# # #
# Replace: revalidation skipped when the leaf layout is unchanged


class TestReplace:
    def _point(self):
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        @strux.struct
        class Point:
            x: Float[Array, "2"]
            y: Float[Array, "2"]

        return Point

    def test_same_layout_replace(self):
        import jax.numpy as jnp
        Point = self._point()
        p = Point(x=jnp.zeros(2), y=jnp.zeros(2))
        q = p.replace(x=jnp.ones(2))
        assert type(q) is type(p)
        assert (q.x == 1.0).all()
        assert (q.y == 0.0).all()
        assert q.shape == ()

    def test_layout_changing_replace_revalidates(self):
        import jax.numpy as jnp
        Point = self._point()
        p = Point(x=jnp.zeros(2), y=jnp.zeros(2))
        # batch both fields: valid, goes through full construction
        q = p.replace(x=jnp.ones((4, 2)), y=jnp.ones((4, 2)))
        assert q.shape == (4,)
        # invalid replacement is caught
        with pytest.raises(strux.ValidationError):
            p.replace(x=jnp.ones(3))
        with pytest.raises(strux.ValidationError):
            p.replace(x=jnp.ones((4, 2)))   # inconsistent with y

    def test_replace_propagates_cache(self):
        import jax.numpy as jnp
        Point = self._point()
        p = Point(x=jnp.ones((4, 2)), y=jnp.ones((4, 2)))
        q = p.replace(x=jnp.zeros((4, 2)))
        assert q.__dict__["_strux_candidates"] == frozenset({(4,)})

    def test_replace_static_field(self):
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        @strux.struct(static_fieldnames=("label",))
        class Tagged:
            x: Float[Array, "2"]
            label: str

        t = Tagged(x=jnp.zeros(2), label="a")
        assert t.replace(label="b").label == "b"

    def test_replace_unknown_field_raises(self):
        import jax.numpy as jnp
        Point = self._point()
        p = Point(x=jnp.zeros(2), y=jnp.zeros(2))
        with pytest.raises(TypeError):
            p.replace(z=jnp.zeros(2))

    def test_post_init_always_runs(self):
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        calls = []

        @strux.struct
        class Counted:
            x: Float[Array, "2"]
            def __post_init__(self):
                calls.append(1)

        c = Counted(x=jnp.zeros(2))
        c.replace(x=jnp.ones(2))
        assert len(calls) == 2

    def test_replace_python_scalar_type_change_revalidates(self):
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        @strux.struct
        class Config:
            u: Float[Array, ""]
            rate: float

        c = Config(u=jnp.float32(1.0), rate=0.5)
        assert type(c.replace(rate=0.25).rate) is float
        # int is not float: the fast path must not treat it as same-layout
        with pytest.raises(strux.ValidationError):
            c.replace(rate=jnp.ones(3))

    def test_replace_on_transformed_instance(self):
        # provisional semantics (functor discussion pending): a replacement
        # with the leaf layout of a tree-transformed instance is allowed,
        # since it leaves the instance exactly as valid as it was
        import jax
        import jax.numpy as jnp
        Point = self._point()
        p = Point(x=jnp.zeros(2), y=jnp.zeros(2))
        mask = jax.tree.map(lambda a: a > 0, p)     # bool leaves, unvalidated
        replaced = mask.replace(x=jnp.array([True, False]))
        assert replaced.x.dtype == jnp.bool
        # a layout-changing replace on such an instance still revalidates
        # (and fails, since bool does not match the float annotation)
        with pytest.raises(strux.ValidationError):
            mask.replace(x=jnp.zeros(3, dtype=bool))
