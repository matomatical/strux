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

    def test_non_pytree_data_field_raises(self):
        @strux.struct
        class Bad:
            pos: Int[Array, "2"]
            name: str

        with pytest.raises(strux.SchemaError, match="not pytree data"):
            Bad["batch"]

    def test_jaxtype_detection_is_exact(self):
        # a class that merely duck-types some jaxtyping attributes is not
        # treated as an array annotation: it is an instance-checked pytree
        # class, and an instance (with no array leaves inside) fails
        # validation at construction
        class Impostor:
            dtype = None
            array_type = None
            dim_str = "2"

        @strux.struct
        class HasImpostor:
            pos: Impostor

        with pytest.raises(strux.ValidationError, match="not an array"):
            HasImpostor(pos=Impostor())

    def test_plain_scalar_hints_promoted(self):
        # plain float/int/bool hints mean "python scalar or array of the
        # matching dtype kind"; batching keeps the array arms
        @strux.struct
        class Metrics:
            pos: Int[Array, "2"]
            loss: float
            step: int
            done: bool

        ann = Metrics["batch"]
        # each scalar field expands to a union of batched array arms
        for name in ("loss", "step", "done"):
            arms = ann._field_hints[name]
            assert isinstance(arms, tuple)
            assert any(getattr(arm, "dim_str", None) == "batch" for arm in arms)
        # a batched instance passes the isinstance check
        batched = Metrics(
            pos=jnp.zeros((4, 2), dtype=jnp.int32),
            loss=jnp.zeros(4),
            step=jnp.zeros(4, dtype=jnp.int32),
            done=jnp.zeros(4, dtype=bool),
        )
        assert isinstance(batched, ann)

    def test_plain_scalar_hint_wrong_dtype_fails(self):
        @strux.struct(check=False)
        class Loss:
            loss: float

        batched = Loss(loss=jnp.zeros(4, dtype=jnp.int32))
        assert not isinstance(batched, Loss["batch"])

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
        # a wrong-dtype instance cannot be constructed under checking, so
        # use an unchecked class to exercise the isinstance path
        @strux.struct(check=False)
        class LooseEnv:
            hero_pos: Int[Array, "2"]

        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.float32),  # wrong dtype
        )
        assert not isinstance(env, LooseEnv["batch"])

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
        # an unbatched child with a batched sibling cannot be constructed
        # under checking, so use unchecked classes for the isinstance path
        @strux.struct(check=False)
        class LooseChild:
            pos: Int[Array, "2"]

        @strux.struct(check=False)
        class LooseWorld:
            env: LooseChild
            score: Float[Array, ""]

        world = LooseWorld(
            env=LooseChild(pos=jnp.ones((2,), dtype=jnp.int32)),  # unbatched
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert not isinstance(world, LooseWorld["batch"])

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
        # such an instance cannot be constructed under checking, so use an
        # unchecked class to exercise the annotation path
        @strux.struct(check=False)
        class LooseEnv:
            hero_pos: Int[Array, "2"]
            goal_pos: Int[Array, "2"]

        @jaxtyped(typechecker=beartype)
        def step(env: LooseEnv["batch"]) -> LooseEnv["batch"]:
            return env

        # hero_pos batch=3, goal_pos batch=4 — inconsistent "batch" dim
        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
        )
        with pytest.raises(Exception):
            step(env)


# # #
# Batch-tolerant constructor checking


class TestCheckedConstruction:
    """
    Construction validates against the schema: dtype kind and trailing
    (element) dims are enforced, leading batch dims are free but must
    agree across fields. The checking is strux's own — the @jaxtyped
    wrapper left on some tests here documents that external checkers
    wrapping the constructor (as jaxtyping's import hook does to every
    class in a checked module) find nothing to enforce and do not
    interfere with batched construction.
    """

    def test_element_construction_passes(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        Goal(weight=jnp.float32(1.0))

    def test_wrong_dtype_raises(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        with pytest.raises(Exception):
            Goal(weight=jnp.int32(1))

    def test_batched_leaves_pass(self):
        # for a rank-0 field every shape is a legal batch shape
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        Goal(weight=jnp.ones((4, 2)))

    def test_trailing_shape_still_enforced(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Path:
            points: Float[Array, "n 2"]

        Path(points=jnp.ones((5, 2)))        # element
        Path(points=jnp.ones((7, 5, 2)))     # batch of 7
        with pytest.raises(Exception):
            Path(points=jnp.ones((5,)))      # rank too low
        with pytest.raises(Exception):
            Path(points=jnp.ones((7, 5, 3))) # trailing dim mismatch

    def test_vmap_construction_passes(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        goals = jax.vmap(lambda w: Goal(weight=w))(jnp.arange(3.0))
        assert goals.shape == (3,)

    def test_scan_construction_passes(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        _, goals = jax.lax.scan(
            lambda carry, w: (carry, Goal(weight=w)),
            None,
            jnp.arange(3.0),
        )
        assert goals.shape == (3,)

    def test_tree_stack_construction_passes(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        goals = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *[Goal(weight=jnp.float32(w)) for w in (0.0, 0.5, 1.0)],
        )
        assert goals.shape == (3,)

    def test_replace_on_batched_struct_passes(self):
        # dataclasses.replace goes through the wrapped __init__ too
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Goal:
            weight: Float[Array, ""]

        goals = jax.vmap(lambda w: Goal(weight=w))(jnp.arange(3.0))
        goals.replace(weight=jnp.ones(3))

    def test_inconsistent_batch_dims_rejected(self):
        # leading batch dims are free, but must agree across array fields
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Pair:
            u: Float[Array, ""]
            v: Float[Array, ""]

        Pair(u=jnp.ones(3), v=jnp.ones(3))
        with pytest.raises(Exception):
            Pair(u=jnp.ones(3), v=jnp.ones(4))
        with pytest.raises(Exception):
            Pair(u=jnp.ones(3), v=jnp.float32(1.0))  # partially batched

    def test_user_dim_named_batch_no_collision(self):
        # a regular dim that the user names "batch" is independent of the
        # "*batch" variadic prefix (separate namespaces in jaxtyping)
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Frames:
            frames: Float[Array, "batch 2"]
            count: Int[Array, ""]

        Frames(frames=jnp.ones((7, 2)), count=jnp.int32(7))
        Frames(
            frames=jnp.ones((3, 7, 2)),
            count=jnp.ones(3, dtype=jnp.int32),
        )

    def test_scalar_field_accepts_scalar_and_batched_array(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Config:
            coeff: float

        Config(coeff=0.5)
        Config(coeff=jnp.ones(3))
        with pytest.raises(Exception):
            Config(coeff=jnp.ones(3, dtype=jnp.int32))

    def test_scalar_field_batch_constraints(self):
        # a python scalar leaves the batch shape unconstrained; an array
        # value participates in the cross-field consistency check
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Config:
            coeff: float
            u: Float[Array, ""]

        Config(coeff=0.5, u=jnp.ones(3))
        Config(coeff=jnp.ones(3), u=jnp.ones(3))
        with pytest.raises(Exception):
            Config(coeff=jnp.ones(4), u=jnp.ones(3))

    def test_static_field_not_runtime_checked(self):
        # static fields are metadata: their annotations are for static
        # checkers, and construction does not validate them
        @strux.struct(static_fieldnames=("label",))
        class Config:
            coeff: Float[Array, ""]
            label: str

        Config(coeff=jnp.float32(1.0), label="ok")
        Config(coeff=jnp.float32(1.0), label=42)    # not rejected

    def test_variadic_annotation_left_alone(self):
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class History:
            values: Float[Array, "..."]

        History(values=jnp.ones((2, 3, 4)))
        with pytest.raises(Exception):
            History(values=jnp.ones((2, 3, 4), dtype=jnp.int32))


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

    def test_plain_scalar_hint_fields(self):
        @strux.struct
        class Metrics:
            loss: float
            step: int

        # python scalars have batch shape ()
        assert Metrics(loss=1.0, step=7).shape == ()
        # a batched instance: all of the value's dims are batch dims
        batched = Metrics(loss=jnp.zeros(4), step=jnp.arange(4))
        assert batched.shape == (4,)

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
        # checked construction refuses inconsistent batch shapes outright
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            )
        # .shape raises the same for an unchecked inconsistent struct
        @strux.struct(check=False)
        class LoosePair:
            u: Float[Array, ""]
            v: Float[Array, ""]

        pair = LoosePair(u=jnp.ones(3), v=jnp.ones(4))
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            pair.shape

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


def _assert_equal(a, b):
    """Assert two arrays have equal values AND dtypes."""
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} != {b.dtype}"
    assert jnp.array_equal(a, b)


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
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_round_trip_dict_tree(self):
        original = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored["a"], original["a"])
        _assert_equal(restored["b"], original["b"])

    def test_round_trip_list_tree(self):
        original = [jnp.array(1.0), jnp.array(2.0)]
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored[0], original[0])
        _assert_equal(restored[1], original[1])

    def test_round_trip_mixed_tree(self):
        original = {"params": _make_env(), "step": jnp.array(0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored["params"].hero_pos, original["params"].hero_pos)
        _assert_equal(restored["step"], original["step"])

    def test_missing_key_raises(self):
        d = {"hero_pos": jnp.zeros(2), "goal_pos": jnp.zeros(2)}
        with pytest.raises(KeyError, match="walls"):
            strux.from_dict(d, template=_make_env())

    def test_extra_keys_raises(self):
        d = strux.to_dict(_make_env())
        d["extra"] = jnp.zeros(3)
        with pytest.raises(KeyError, match="extra keys"):
            strux.from_dict(d, template=_make_env())

    def test_static_fields_from_template(self):
        @strux.struct(static_fieldnames=("label",))
        class Labelled:
            pos: Int[Array, "2"]
            label: str
        template = Labelled(pos=jnp.zeros(2, dtype=jnp.int32), label="hello")
        d = {"pos": jnp.array([10, 20], dtype=jnp.int32)}
        restored = strux.from_dict(d, template=template)
        _assert_equal(restored.pos, jnp.array([10, 20]))
        assert restored.label == "hello"


class TestSaveLoadNpz:
    def test_flat_struct(self, tmp_path):
        original = _make_env()
        path = tmp_path / "env.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Environment)
        _assert_equal(restored.hero_pos, original.hero_pos)
        _assert_equal(restored.walls, original.walls)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.walls, original.env.walls)


class TestSaveLoadSafetensors:
    def test_flat_struct(self, tmp_path):
        original = Point(x=jnp.float32(1.0), y=jnp.float32(2.0))
        path = tmp_path / "point.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Point)
        _assert_equal(restored.x, original.x)
        _assert_equal(restored.y, original.y)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)


class TestSaveLoadErrors:
    def test_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot infer format"):
            strux.save(tmp_path / "file.xyz", _make_env())

    def test_explicit_format_overrides_extension(self, tmp_path):
        path = tmp_path / "file.npz"
        strux.save(path, _make_env(), fmt="savez")
        restored = strux.load(path, template=_make_env(), fmt="savez")
        _assert_equal(restored.hero_pos, _make_env().hero_pos)

    def test_overwrite_raises(self, tmp_path):
        path = tmp_path / "env.npz"
        strux.save(path, _make_env())
        with pytest.raises(FileExistsError, match="already exists"):
            strux.save(path, _make_env())

    def test_overwrite_true_replaces(self, tmp_path):
        path = tmp_path / "env.npz"
        original = _make_env()
        strux.save(path, original)
        updated = original.replace(hero_pos=original.hero_pos + 1)
        strux.save(path, updated, overwrite=True)
        restored = strux.load(path, template=original)
        _assert_equal(restored.hero_pos, updated.hero_pos)

    def test_no_temporary_files_left_behind(self, tmp_path):
        # saves go via a temporary file renamed over the destination
        path = tmp_path / "env.npz"
        strux.save(path, _make_env())
        strux.save(path, _make_env(), overwrite=True)
        assert [p.name for p in tmp_path.iterdir()] == ["env.npz"]

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
        strux.save(path_explicit, big, fmt="savez_compressed")  # explicit
        strux.save(path_uncompressed, big, fmt="savez")         # uncompressed
        # all three round-trip correctly
        for p in (path_default, path_explicit, path_uncompressed):
            restored = strux.load(p, template=big)
            _assert_equal(restored.data, big.data)
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
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_save_method_overwrite(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        original.save(path)
        original.save(path, overwrite=True)
        restored = original.restore(path)
        _assert_equal(restored.score, original.score)

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
# Schema-driven construction and batch inference


class TestSchemaSolver:
    def test_unannotated_array_pinned_by_sibling(self):
        # a bare jax.Array field gives only a prefix constraint; a
        # rank-determined sibling pins the batch and the bare field
        # participates fully
        @strux.struct
        class WithAux:
            walls: Bool[Array, "h w"]
            aux: jax.Array

        unbatched = WithAux(walls=jnp.zeros((5, 5), bool), aux=jnp.zeros((7, 3)))
        assert unbatched.shape == ()
        batched = WithAux(
            walls=jnp.zeros((4, 5, 5), bool),
            aux=jnp.zeros((4, 7, 3)),
        )
        assert batched.shape == (4,)
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            WithAux(walls=jnp.zeros((4, 5, 5), bool), aux=jnp.zeros((3, 7)))

    def test_underdetermined_shape_raises(self):
        @strux.struct
        class Blob:
            data: Float[Array, "..."]

        blob = Blob(data=jnp.zeros((2, 3)))
        with pytest.raises(ValueError, match="under-determined"):
            blob.shape

    def test_optional_field(self):
        @strux.struct
        class Opt:
            momentum: Float[Array, "n"] | None
            count: Int[Array, ""]

        assert Opt(momentum=None, count=jnp.int32(0)).shape == ()
        assert Opt(momentum=jnp.zeros(5), count=jnp.int32(0)).shape == ()
        batched = Opt(momentum=jnp.zeros((8, 5)), count=jnp.zeros(8, jnp.int32))
        assert batched.shape == (8,)
        with pytest.raises(strux.ValidationError, match="no arm"):
            Opt(momentum="hello", count=jnp.int32(0))

    def test_rank_ambiguous_union_disambiguated_by_sibling(self):
        @strux.struct
        class Amb:
            history: Float[Array, "n"] | Float[Array, "n m"]
            anchor: Int[Array, ""]

        # same history value, arm decided by the anchor's batch
        arm1 = Amb(history=jnp.zeros((32, 5)), anchor=jnp.zeros(32, jnp.int32))
        assert arm1.shape == (32,)
        arm2 = Amb(history=jnp.zeros((32, 5)), anchor=jnp.int32(0))
        assert arm2.shape == ()

    def test_arraylike_scalar_field_all_features(self):
        # the annotation style that motivated the schema: works through
        # construction, .shape, and batched subscripting alike
        import jax.typing

        @strux.struct
        class Scaled:
            coeff: Float[jax.typing.ArrayLike, ""]
            walls: Float[Array, "h w"]

        s = Scaled(coeff=0.5, walls=jnp.zeros((5, 5)))
        assert s.shape == ()
        b = Scaled(coeff=jnp.ones(32), walls=jnp.zeros((32, 5, 5)))
        assert b.shape == (32,)
        assert isinstance(b, Scaled["batch"])
        assert not isinstance(s, Scaled["batch"])
        with pytest.raises(strux.ValidationError):
            Scaled(coeff=jnp.zeros(3, dtype=int), walls=jnp.zeros((3, 5, 5)))

    def test_container_fields(self):
        @strux.struct
        class Bank:
            layers: tuple[Float[Array, "n"], ...]
            table: dict[str, Float[Array, "2"]]

        bank = Bank(
            layers=(jnp.zeros(3), jnp.zeros(5)),
            table={"a": jnp.zeros(2), "b": jnp.ones(2)},
        )
        assert bank.shape == ()
        batched = Bank(
            layers=(jnp.zeros((4, 3)), jnp.zeros((4, 5))),
            table={"a": jnp.zeros((4, 2))},
        )
        assert batched.shape == (4,)
        assert isinstance(batched, Bank["batch"])
        assert not isinstance(bank, Bank["batch"])
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            Bank(
                layers=(jnp.zeros((4, 3)), jnp.zeros((3, 5))),
                table={},
            )
        with pytest.raises(strux.ValidationError):
            Bank(layers=(jnp.zeros(3),), table={"a": jnp.zeros(7)})

    def test_fixed_tuple_field(self):
        @strux.struct
        class Pair:
            bounds: tuple[Float[Array, ""], Int[Array, "2"]]

        Pair(bounds=(jnp.float32(0.0), jnp.zeros(2, jnp.int32)))
        with pytest.raises(strux.ValidationError, match="length"):
            Pair(bounds=(jnp.float32(0.0),))

    def test_polymorphic_class_field(self):
        # an abstract (non-struct) base annotation: instances are checked
        # by isinstance and recursed via their own dynamic type
        class RewardFn:
            pass

        @strux.struct
        class Constant(RewardFn):
            value: Float[Array, ""]

        @strux.struct
        class Sum(RewardFn):
            terms: tuple[RewardFn, ...]

        s = Sum(terms=(Constant(value=jnp.float32(1.0)),))
        assert s.shape == ()
        batched = Sum(terms=(Constant(value=jnp.zeros(4)),))
        assert batched.shape == (4,)
        assert isinstance(batched, Sum["batch"])
        with pytest.raises(strux.ValidationError, match="expected an instance"):
            Sum(terms=(jnp.zeros(3),))

    def test_base_annotation_subclass_value(self):
        # a field annotated with a struct base class holding a subclass
        # instance validates against the subclass's own schema
        @strux.struct
        class Base:
            steps: Int[Array, ""]

        @strux.struct
        class Derived(Base):
            walls: Bool[Array, "h w"]

        @strux.struct
        class Holder:
            env: Base

        d = Derived(steps=jnp.int32(0), walls=jnp.zeros((5, 5), bool))
        assert Holder(env=d).shape == ()
        db = Derived(
            steps=jnp.zeros(4, jnp.int32),
            walls=jnp.zeros((4, 5, 5), bool),
        )
        assert Holder(env=db).shape == (4,)
        assert isinstance(Holder(env=db), Holder["batch"])

    def test_python_scalars_are_batch_agnostic(self):
        @strux.struct
        class Config:
            coeff: float
            u: Float[Array, ""]

        assert Config(coeff=0.5, u=jnp.ones(3)).shape == (3,)
        with pytest.raises(strux.ValidationError):
            Config(coeff=jnp.ones(4), u=jnp.ones(3))

    def test_vjp_through_struct_with_int_field(self):
        # autodiff produces float0 cotangents for integer leaves; the
        # cotangent struct must remain constructible
        @strux.struct
        class Mixed:
            w: Float[Array, "3"]
            count: Int[Array, ""]

        def loss(m):
            return jnp.sum(m.w ** 2)

        m = Mixed(w=jnp.ones(3), count=jnp.int32(2))
        _, vjp_fn = jax.vjp(loss, m)
        (g,) = vjp_fn(1.0)
        assert g.w.shape == (3,)
        assert g.count.dtype == jax.dtypes.float0

    def test_init_signature_carries_no_annotations(self):
        # external checkers wrapping the constructor must find nothing to
        # enforce, whether they read __annotations__ or inspect.signature
        import inspect

        assert Environment.__init__.__annotations__ in ({}, None) or all(
            a is inspect.Parameter.empty
            for a in Environment.__init__.__annotations__.values()
        )
        signature = inspect.signature(Environment.__init__)
        assert all(
            p.annotation is inspect.Parameter.empty
            for p in signature.parameters.values()
        )


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
