import warnings

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype

import strux


# # # 
# Common example test structs (other bespoke ones defined inline)


@strux.struct
class Point:
    x: Float[Array, ""]
    y: Float[Array, ""]


@strux.struct
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "h w"]


@strux.struct
class World:
    env: Environment
    score: Float[Array, ""]


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
# Pretty printing (to_str)


class TestToStr:
    # scalars
    def test_int(self):
        assert strux.to_str(42) == "int(42)"

    def test_float(self):
        assert strux.to_str(1.5) == "float(1.5)"

    def test_bool(self):
        assert strux.to_str(True) == "bool(True)"

    def test_complex(self):
        assert strux.to_str(1+2j) == "complex((1+2j))"

    def test_str(self):
        assert strux.to_str("hello") == "str('hello')"

    def test_none(self):
        assert strux.to_str(None) == "None"

    # arrays
    def test_jnp_scalar(self):
        assert strux.to_str(jnp.float32(1.0)) == "jnp.float32[]"

    def test_jnp_1d(self):
        assert strux.to_str(jnp.zeros(3)) == "jnp.float32[3]"

    def test_jnp_2d(self):
        assert strux.to_str(jnp.zeros((3, 4))) == "jnp.float32[3,4]"

    def test_np_1d(self):
        import numpy as np
        assert strux.to_str(np.zeros(3)) == "np.float64[3]"

    def test_np_2d(self):
        import numpy as np
        assert strux.to_str(np.zeros((3, 4))) == "np.float64[3,4]"

    # containers
    def test_tuple(self):
        result = strux.to_str((1, 2.0))
        assert result == "(\n  int(1),\n  float(2.0),\n)"

    def test_list(self):
        result = strux.to_str([1, 2.0])
        assert result == "[\n  int(1),\n  float(2.0),\n]"

    def test_dict(self):
        result = strux.to_str({"a": 1})
        assert result == "{\n  'a': int(1),\n}"

    def test_empty_tuple(self):
        assert strux.to_str(()) == "(\n)"

    def test_empty_list(self):
        assert strux.to_str([]) == "[\n]"

    def test_empty_dict(self):
        assert strux.to_str({}) == "{\n}"

    # namedtuples
    def test_namedtuple(self):
        from collections import namedtuple
        Pair = namedtuple("Pair", ["x", "y"])
        result = strux.to_str(Pair(1.0, 2.0))
        assert result == "Pair(\n  x=float(1.0),\n  y=float(2.0),\n)"

    def test_namedtuple_max_depth(self):
        from collections import namedtuple
        Pair = namedtuple("Pair", ["x", "y"])
        assert strux.to_str(Pair(1.0, 2.0), max_depth=0) == "Pair(...)"

    # callables
    def test_callable_with_name(self):
        assert strux.to_str(jax.nn.relu) == "<fn:relu>"

    def test_callable_without_name(self):
        import functools
        p = functools.partial(int, base=2)
        result = strux.to_str(p)
        assert result.startswith("functools.partial(")

    # structs
    def test_struct(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        result = strux.to_str(env)
        assert result.startswith("Environment(\n")
        assert "hero_pos=jnp.int32[2]," in result
        assert "walls=jnp.bool[5,5]," in result

    def test_struct_max_depth(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert strux.to_str(env, max_depth=0) == "Environment(...)"

    # nested
    def test_nested_struct(self):
        world = World(
            env=Environment(
                hero_pos=jnp.array([1, 2], dtype=jnp.int32),
                goal_pos=jnp.array([3, 4], dtype=jnp.int32),
                walls=jnp.zeros((5, 5), dtype=bool),
            ),
            score=jnp.float32(1.0),
        )
        result = strux.to_str(world)
        assert "World(" in result
        assert "  env=Environment(" in result
        assert "    hero_pos=jnp.int32[2]," in result

    def test_nested_max_depth_1(self):
        world = World(
            env=Environment(
                hero_pos=jnp.array([1, 2], dtype=jnp.int32),
                goal_pos=jnp.array([3, 4], dtype=jnp.int32),
                walls=jnp.zeros((5, 5), dtype=bool),
            ),
            score=jnp.float32(1.0),
        )
        result = strux.to_str(world, max_depth=1)
        assert "env=Environment(...)," in result
        assert "score=jnp.float32[]," in result

    # custom indent
    def test_custom_indent(self):
        result = strux.to_str((1,), indent="\t")
        assert result == "(\n\tint(1),\n)"

    # unknown leaf
    def test_unknown_leaf(self):
        result = strux.to_str(object())
        assert result.startswith("UNKNOWN_LEAF:")

    # custom pytree class (not a dataclass)
    def test_custom_pytree_class(self):
        class MyNode:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        jax.tree_util.register_pytree_node(
            MyNode,
            lambda n: ((n.x, n.y), None),
            lambda _, children: MyNode(*children),
        )
        result = strux.to_str(MyNode(1.0, 2.0))
        assert "UNKNOWN_LEAF" in result


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
# Annotation expansion (static behaviour)


class TestAnnotationExpansion:
    def test_jaxtyping_fields_prepended(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch 2"
        assert hints["goal_pos"].dim_str == "batch 2"
        assert hints["walls"].dim_str == "batch h w"

    def test_preserves_dtype_class(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        # dtype should point back to the original jaxtyping dtype class
        assert hints["hero_pos"].dtype is Int
        assert hints["walls"].dtype is Bool

    def test_preserves_array_type(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        for hint in hints.values():
            assert hint.array_type is jax.Array

    def test_nested_struct_recursion(self):
        ann = World["batch"]
        hints = ann._field_hints
        # env field should be an expanded struct annotation
        env_ann = hints["env"]
        assert env_ann._struct_type is Environment
        assert env_ann._field_hints["hero_pos"].dim_str == "batch 2"
        assert env_ann._field_hints["walls"].dim_str == "batch h w"
        # score field should be directly expanded
        assert hints["score"].dim_str == "batch"

    def test_meta_fields_skipped(self):
        @strux.struct(static_fieldnames=("name",))
        class WithMeta:
            pos: Int[Array, "2"]
            name: str
        ann = WithMeta["batch"]
        hints = ann._field_hints
        # jaxtyping data field is expanded
        assert hints["pos"].dim_str == "batch 2"
        # meta field is not included
        assert "name" not in hints

    def test_non_batchable_data_field_raises(self):
        @strux.struct
        class Bad:
            pos: Int[Array, "2"]
            name: str

        with pytest.raises(TypeError, match="Cannot batch data field 'name'"):
            Bad["batch"]

    def test_scalar_type_hint_in_error(self):
        @strux.struct
        class Bad:
            pos: Int[Array, "2"]
            loss: float

        with pytest.raises(TypeError, match='Float\\[Array, ""\\]'):
            Bad["batch"]

    def test_scalar_fields_no_trailing_space(self):
        # Float[Array, ""] is a scalar; batching should give "batch", not "batch "
        ann = Point["batch"]
        assert ann._field_hints["x"].dim_str == "batch"
        assert ann._field_hints["y"].dim_str == "batch"

    def test_empty_dims_is_noop(self):
        ann = Environment[""]
        assert ann._field_hints["hero_pos"].dim_str == "2"
        assert ann._field_hints["walls"].dim_str == "h w"

    def test_multi_dims(self):
        ann = Environment["batch time"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch time 2"
        assert hints["walls"].dim_str == "batch time h w"

    def test_caching(self):
        a = Environment["batch"]
        b = Environment["batch"]
        assert a is b

    def test_different_dims_not_cached_together(self):
        a = Environment["batch"]
        b = Environment["time"]
        assert a is not b

    def test_annotation_name(self):
        ann = Environment["batch"]
        assert ann.__name__ == 'Environment["batch"]'


# # # 
# Runtime isinstance checks


class TestInstanceCheck:
    def test_base_type_still_works(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert isinstance(env, Environment)

    def test_unbatched_fails_batched_annotation(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert not isinstance(env, Environment["batch"])

    def test_batched_passes(self):
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        assert isinstance(env, Environment["batch"])

    def test_wrong_dtype_fails(self):
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.float32),  # wrong dtype
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        assert not isinstance(env, Environment["batch"])

    def test_wrong_type_fails(self):
        assert not isinstance("not an env", Environment["batch"])
        assert not isinstance(42, Environment["batch"])

    def test_nested_struct_passes(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(world, World["batch"])

    def test_nested_struct_fails_if_child_wrong(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((2,), dtype=jnp.int32),  # unbatched
                goal_pos=jnp.ones((2,), dtype=jnp.int32),
                walls=jnp.zeros((5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert not isinstance(world, World["batch"])

    def test_meta_field_not_checked(self):
        @strux.struct(static_fieldnames=("name",))
        class WithMeta:
            pos: Int[Array, "2"]
            name: str
        obj = WithMeta(
            pos=jnp.ones((3, 2), dtype=jnp.int32),
            name="hello",
        )
        # check that meta field (name) is not checked during isinstance
        assert isinstance(obj, WithMeta["batch"])

    def test_scalar_struct_batched(self):
        point = Point(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(point, Point["batch"])


# # # 
# Integration with jaxtyping + beartype runtime type checking


class TestJaxtypedIntegration:
    def test_correct_annotation_passes(self):
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        result = step(env)
        assert isinstance(result, Environment)

    def test_wrong_annotation_raises(self):
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),  # unbatched
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        with pytest.raises(Exception):
            step(env)

    def test_nested_struct_jaxtyped(self):
        @jaxtyped(typechecker=beartype)
        def step(world: World["batch"]) -> World["batch"]:
            return world

        world = World(
            env=Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        result = step(world)
        assert isinstance(result, World)

    def test_dimension_consistency_checked(self):
        """Within a @jaxtyped context, named dims must be consistent."""
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        # hero_pos batch=3, goal_pos batch=4 — inconsistent "batch" dim
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        with pytest.raises(Exception):
            step(env)


# # #
# Batch shape


class TestShape:
    def test_unbatched(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert env.shape == ()

    def test_single_batch_dim(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        assert env.shape == (4,)

    def test_multi_batch_dims(self):
        env = Environment(
            hero_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 3, 5, 5), dtype=bool),
        )
        assert env.shape == (4, 3)

    def test_scalar_fields(self):
        p = Point(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]))
        assert p.shape == (2,)

    def test_nested_struct(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((4, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0, 4.0]),
        )
        assert world.shape == (4,)

    def test_inconsistent_batch_raises(self):
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        with pytest.raises(ValueError, match="Inconsistent batch shapes"):
            env.shape

    def test_shape_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'shape'"):
            @strux.struct
            class HasShape:
                shape: int
                x: float


# # #
# Indexing


class TestGetitem:
    def test_integer_index(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[0]
        assert isinstance(e, Environment)
        assert e.hero_pos.shape == (2,)
        assert e.walls.shape == (5, 5)

    def test_slice(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[1:3]
        assert isinstance(e, Environment)
        assert e.hero_pos.shape == (2, 2)
        assert e.walls.shape == (2, 5, 5)

    def test_multi_batch_index(self):
        env = Environment(
            hero_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 3, 5, 5), dtype=bool),
        )
        e = env[0]
        assert e.hero_pos.shape == (3, 2)
        assert e.walls.shape == (3, 5, 5)
        e2 = env[0, 1]
        assert e2.hero_pos.shape == (2,)
        assert e2.walls.shape == (5, 5)

    def test_advanced_indexing(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[jnp.array([0, 2])]
        assert e.hero_pos.shape == (2, 2)
        assert e.walls.shape == (2, 5, 5)

    def test_nested_struct_index(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((4, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0, 4.0]),
        )
        w = world[0]
        assert isinstance(w, World)
        assert isinstance(w.env, Environment)
        assert w.env.hero_pos.shape == (2,)
        assert w.score.shape == ()

    def test_index_preserves_values(self):
        env = Environment(
            hero_pos=jnp.array([[10, 20], [30, 40]], dtype=jnp.int32),
            goal_pos=jnp.array([[50, 60], [70, 80]], dtype=jnp.int32),
            walls=jnp.zeros((2, 5, 5), dtype=bool),
        )
        e = env[1]
        assert jnp.array_equal(e.hero_pos, jnp.array([30, 40]))
        assert jnp.array_equal(e.goal_pos, jnp.array([70, 80]))


# # #
# Serialisation


def _make_env():
    return Environment(
        hero_pos=jnp.array([1, 2], dtype=jnp.int32),
        goal_pos=jnp.array([3, 4], dtype=jnp.int32),
        walls=jnp.ones((5, 5), dtype=bool),
    )


def _make_world():
    return World(env=_make_env(), score=jnp.float32(42.0))


class TestToDict:
    def test_flat_struct_keys(self):
        d = strux.to_dict(_make_env())
        assert set(d.keys()) == {"hero_pos", "goal_pos", "walls"}

    def test_nested_struct_keys(self):
        d = strux.to_dict(_make_world())
        assert set(d.keys()) == {"env/hero_pos", "env/goal_pos", "env/walls", "score"}

    def test_values_are_numpy(self):
        import numpy
        d = strux.to_dict(_make_env())
        for v in d.values():
            assert isinstance(v, numpy.ndarray)

    def test_dict_keys_use_repr(self):
        tree = {"a/b": jnp.array(1.0), "c": jnp.array(2.0)}
        d = strux.to_dict(tree)
        assert set(d.keys()) == {"'a/b'", "'c'"}

    def test_sequence_keys_use_repr(self):
        tree = [jnp.array(1.0), jnp.array(2.0)]
        d = strux.to_dict(tree)
        assert set(d.keys()) == {"0", "1"}

    def test_mixed_tree_keys(self):
        tree = {"params": _make_env(), "steps": [jnp.array(1), jnp.array(2)]}
        d = strux.to_dict(tree)
        assert "'params'/hero_pos" in d
        assert "'steps'/0" in d

    def test_key_clash_raises(self):
        class Evil(str):
            def __repr__(self):
                return "'a'"
        tree = {"a": jnp.array(1.0), Evil("b"): jnp.array(2.0)}
        with pytest.raises(ValueError, match="Key clash"):
            strux.to_dict(tree)


class TestFromDict:
    def test_round_trip(self):
        original = _make_world()
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        assert jnp.array_equal(restored.score, original.score)
        assert jnp.array_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_round_trip_dict_tree(self):
        original = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        assert jnp.array_equal(restored["a"], original["a"])
        assert jnp.array_equal(restored["b"], original["b"])

    def test_round_trip_list_tree(self):
        original = [jnp.array(1.0), jnp.array(2.0)]
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        assert jnp.array_equal(restored[0], original[0])
        assert jnp.array_equal(restored[1], original[1])

    def test_round_trip_mixed_tree(self):
        original = {"params": _make_env(), "step": jnp.array(0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        assert jnp.array_equal(restored["params"].hero_pos, original["params"].hero_pos)
        assert jnp.array_equal(restored["step"], original["step"])

    def test_missing_key_raises(self):
        d = {"hero_pos": jnp.zeros(2), "goal_pos": jnp.zeros(2)}
        with pytest.raises(KeyError, match="walls"):
            strux.from_dict(d, template=_make_env())

    def test_extra_keys_ok(self):
        d = strux.to_dict(_make_env())
        d["extra"] = jnp.zeros(3)
        restored = strux.from_dict(d, template=_make_env())
        assert jnp.array_equal(restored.hero_pos, _make_env().hero_pos)

    def test_static_fields_from_template(self):
        @strux.struct(static_fieldnames=("label",))
        class Labelled:
            pos: Int[Array, "2"]
            label: str
        template = Labelled(pos=jnp.zeros(2, dtype=jnp.int32), label="hello")
        d = {"pos": jnp.array([10, 20], dtype=jnp.int32)}
        restored = strux.from_dict(d, template=template)
        assert jnp.array_equal(restored.pos, jnp.array([10, 20]))
        assert restored.label == "hello"


class TestSaveLoadNpz:
    def test_flat_struct(self, tmp_path):
        original = _make_env()
        path = tmp_path / "env.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Environment)
        assert jnp.array_equal(restored.hero_pos, original.hero_pos)
        assert jnp.array_equal(restored.walls, original.walls)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        assert jnp.array_equal(restored.score, original.score)
        assert jnp.array_equal(restored.env.walls, original.env.walls)


class TestSaveLoadSafetensors:
    def test_flat_struct(self, tmp_path):
        original = Point(x=jnp.float32(1.0), y=jnp.float32(2.0))
        path = tmp_path / "point.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Point)
        assert jnp.array_equal(restored.x, original.x)
        assert jnp.array_equal(restored.y, original.y)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        assert jnp.array_equal(restored.score, original.score)
        assert jnp.array_equal(restored.env.hero_pos, original.env.hero_pos)


class TestSaveLoadErrors:
    def test_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot infer format"):
            strux.save(tmp_path / "file.xyz", _make_env())

    def test_explicit_format_overrides_extension(self, tmp_path):
        path = tmp_path / "file.npz"
        strux.save(path, _make_env(), format="savez")
        restored = strux.load(path, template=_make_env(), format="savez")
        assert jnp.array_equal(restored.hero_pos, _make_env().hero_pos)

    def test_npz_defaults_to_compressed(self, tmp_path):
        # use a large zero array so compression is clearly effective
        @strux.struct
        class Big:
            data: Float[Array, "n"]
        big = Big(data=jnp.zeros(10_000))
        path_default = tmp_path / "default.npz"
        path_explicit = tmp_path / "explicit.npz"
        path_uncompressed = tmp_path / "uncompressed.npz"
        strux.save(path_default, big)                              # default
        strux.save(path_explicit, big, format="savez_compressed")  # explicit
        strux.save(path_uncompressed, big, format="savez")         # uncompressed
        # all three round-trip correctly
        for p in (path_default, path_explicit, path_uncompressed):
            restored = strux.load(p, template=big)
            assert jnp.array_equal(restored.data, big.data)
        # default and explicit compressed produce the same file size
        assert path_default.stat().st_size == path_explicit.stat().st_size
        # compressed is strictly smaller than uncompressed
        assert path_default.stat().st_size < path_uncompressed.stat().st_size


class TestSaveRestoreMethods:
    def test_save_and_restore(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        original.save(path)
        restored = original.restore(path)
        assert isinstance(restored, World)
        assert jnp.array_equal(restored.score, original.score)
        assert jnp.array_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_save_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'save'"):
            @strux.struct
            class HasSave:
                save: int
                x: float

    def test_restore_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'restore'"):
            @strux.struct
            class HasRestore:
                restore: int
                x: float

