"""
Tests for the Struct type form (strux.Struct[Cls, ...]): static expansion,
functor images, runtime isinstance checks, and integration with jaxtyping +
beartype.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype

import strux

from example_structs import Environment, Point, World


# # #
# Annotation expansion (static behaviour)


class TestAnnotationExpansion:
    def test_jaxtyping_fields_prepended(self):
        ann = strux.Struct[Environment, "batch"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch 2"
        assert hints["goal_pos"].dim_str == "batch 2"
        assert hints["walls"].dim_str == "batch h w"

    def test_preserves_dtype_class(self):
        ann = strux.Struct[Environment, "batch"]
        hints = ann._field_hints
        # dtype should point back to the original jaxtyping dtype class
        assert hints["hero_pos"].dtype is Int
        assert hints["walls"].dtype is Bool

    def test_preserves_array_type(self):
        ann = strux.Struct[Environment, "batch"]
        hints = ann._field_hints
        for hint in hints.values():
            assert hint.array_type is jax.Array

    def test_nested_struct_recursion(self):
        ann = strux.Struct[World, "batch"]
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
        ann = strux.Struct[WithMeta, "batch"]
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
            strux.Struct[Bad, "batch"]

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

        ann = strux.Struct[Metrics, "batch"]
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
        assert not isinstance(batched, strux.Struct[Loss, "batch"])

    def test_scalar_fields_no_trailing_space(self):
        # Float[Array, ""] is a scalar; batching should give "batch", not "batch "
        ann = strux.Struct[Point, "batch"]
        assert ann._field_hints["x"].dim_str == "batch"
        assert ann._field_hints["y"].dim_str == "batch"

    def test_empty_dims_is_rank_exact(self):
        ann = strux.Struct[Environment, ""]
        assert ann._field_hints["hero_pos"].dim_str == "2"
        assert ann._field_hints["walls"].dim_str == "h w"

    def test_multi_dims(self):
        ann = strux.Struct[Environment, "batch time"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch time 2"
        assert hints["walls"].dim_str == "batch time h w"

    def test_chained_prepends_apply_left_to_right(self):
        # "time" is applied first, then "batch" in front of it
        ann = strux.Struct[Environment, "time", "batch"]
        assert ann._field_hints["hero_pos"].dim_str == "batch time 2"

    def test_caching(self):
        a = strux.Struct[Environment, "batch"]
        b = strux.Struct[Environment, "batch"]
        assert a is b

    def test_different_dims_not_cached_together(self):
        a = strux.Struct[Environment, "batch"]
        b = strux.Struct[Environment, "time"]
        assert a is not b

    def test_annotation_name(self):
        ann = strux.Struct[Environment, "batch"]
        assert ann.__name__ == "Struct[Environment, 'batch']"


# # #
# The form's argument handling


class TestStructFormArguments:
    def test_class_subscripting_is_gone(self):
        # the breaking change: structs no longer overload subscripts —
        # subscripting a non-generic struct is a plain TypeError
        with pytest.raises(TypeError):
            Environment["batch"]

    def test_generic_structs_keep_typing_subscripts(self):
        @strux.struct
        class Box[T]:
            item: T | None

        alias = Box[Environment]
        import typing
        assert typing.get_origin(alias) is Box

    def test_single_argument_rejected(self):
        with pytest.raises(TypeError, match="at least one"):
            strux.Struct[Environment]

    def test_non_dataclass_rejected(self):
        with pytest.raises(TypeError, match="dataclass"):
            strux.Struct[42, "batch"]

    def test_bad_functor_rejected(self):
        with pytest.raises(TypeError, match="dims strings or functors"):
            strux.Struct[Environment, 42]

    def test_form_not_instantiable(self):
        from strux.annotate import Struct as RuntimeStruct
        with pytest.raises(TypeError, match="type form"):
            RuntimeStruct()

    def test_generic_alias_target_unwrapped(self):
        # Struct[Box[Environment], "b"]: parameters are erased at runtime,
        # the check is against the origin class
        @strux.struct
        class Box[T: Environment]:
            item: T | None

        ann = strux.Struct[Box[Environment], "b"]
        assert ann._struct_type is Box

    def test_bad_scalar_kind_rejected(self):
        with pytest.raises(TypeError, match="scalar class"):
            strux.astype(str)
        with pytest.raises(TypeError, match="scalar class"):
            strux.mapped(list)


# # #
# Functor images


class TestFunctorImages:
    def _env(self, batched=False):
        shape = (4,) if batched else ()
        return Environment(
            hero_pos=jnp.ones((*shape, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((*shape, 2), dtype=jnp.int32),
            walls=jnp.zeros((*shape, 5, 5), dtype=bool),
        )

    def test_astype_keeps_dims(self):
        ann = strux.Struct[Environment, strux.astype(bool)]
        assert ann._field_hints["hero_pos"].dim_str == "2"
        assert ann._field_hints["hero_pos"].dtype is Bool
        assert ann._field_hints["walls"].dim_str == "h w"

    def test_astype_matches_elementwise_compare_image(self):
        env = self._env()
        image = jax.tree.map(lambda a, b: a == b, env, env)
        assert isinstance(image, strux.Struct[Environment, strux.astype(bool)])
        # and the original does not satisfy the image type
        assert not isinstance(env, strux.Struct[Environment, strux.astype(bool)])

    def test_mapped_matches_full_reduction_image(self):
        env = self._env()
        image = jax.tree.map(jnp.array_equal, env, env)
        assert isinstance(image, strux.Struct[Environment, strux.mapped(bool)])
        counts = jax.tree.map(jnp.count_nonzero, env)
        assert isinstance(counts, strux.Struct[Environment, strux.mapped(int)])
        assert not isinstance(counts, strux.Struct[Environment, strux.mapped(bool)])

    def test_mapped_recurses_nested_structs(self):
        world = World(env=self._env(), score=jnp.zeros(()))
        image = jax.tree.map(jnp.array_equal, world, world)
        assert isinstance(image, strux.Struct[World, strux.mapped(bool)])

    def test_mapped_then_prepend_is_vmapped_reduction(self):
        batched = self._env(batched=True)
        vimage = jax.vmap(
            lambda a, b: jax.tree.map(jnp.array_equal, a, b)
        )(batched, batched)
        ann = strux.Struct[Environment, strux.mapped(bool), "b"]
        assert isinstance(vimage, ann)
        # the unbatched reduction image does not carry the batch dim
        unbatched_image = jax.tree.map(
            jnp.array_equal, self._env(), self._env(),
        )
        assert not isinstance(unbatched_image, ann)

    def test_mapped_absorbs_earlier_functors(self):
        # a full reduction of a batched struct eats the batch dim too:
        # Struct[Env, "b", mapped(bool)] and Struct[Env, mapped(bool)]
        # describe the same image
        batched = self._env(batched=True)
        image = jax.tree.map(jnp.array_equal, batched, batched)
        assert isinstance(image, strux.Struct[Environment, "b", strux.mapped(bool)])
        assert isinstance(image, strux.Struct[Environment, strux.mapped(bool)])

    def test_astype_then_prepend(self):
        batched = self._env(batched=True)
        image = jax.tree.map(lambda a, b: a == b, batched, batched)
        assert isinstance(
            image, strux.Struct[Environment, strux.astype(bool), "b"],
        )

    def test_mapped_scalar_sugar(self):
        # the mapped image accepts python scalars as well as rank-0 arrays,
        # like a plain scalar field annotation
        @strux.struct(check=False)
        class Loose:
            x: Float[Array, "n"]

        assert isinstance(Loose(x=True), strux.Struct[Loose, strux.mapped(bool)])
        assert isinstance(
            Loose(x=jnp.array(True)), strux.Struct[Loose, strux.mapped(bool)],
        )

    def test_functor_repr(self):
        assert repr(strux.astype(bool)) == "astype(bool)"
        assert repr(strux.mapped(int)) == "mapped(int)"


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
        assert not isinstance(env, strux.Struct[Environment, "batch"])

    def test_batched_passes(self):
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        assert isinstance(env, strux.Struct[Environment, "batch"])

    def test_any_rank_dims(self):
        unbatched = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        batched = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        ann = strux.Struct[Environment, "..."]
        assert isinstance(unbatched, ann)
        assert isinstance(batched, ann)

    def test_rank_exact_dims(self):
        unbatched = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        batched = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        ann = strux.Struct[Environment, ""]
        assert isinstance(unbatched, ann)
        assert not isinstance(batched, ann)

    def test_wrong_dtype_fails(self):
        # a wrong-dtype instance cannot be constructed under checking, so
        # use an unchecked class to exercise the isinstance path
        @strux.struct(check=False)
        class LooseEnv:
            hero_pos: Int[Array, "2"]

        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.float32),  # wrong dtype
        )
        assert not isinstance(env, strux.Struct[LooseEnv, "batch"])

    def test_wrong_type_fails(self):
        assert not isinstance("not an env", strux.Struct[Environment, "batch"])
        assert not isinstance(42, strux.Struct[Environment, "batch"])

    def test_nested_struct_passes(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(world, strux.Struct[World, "batch"])

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
        assert not isinstance(world, strux.Struct[LooseWorld, "batch"])

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
        assert isinstance(obj, strux.Struct[WithMeta, "batch"])

    def test_scalar_struct_batched(self):
        point = Point(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(point, strux.Struct[Point, "batch"])


# # #
# Integration with jaxtyping + beartype runtime type checking


class TestJaxtypedIntegration:
    def test_correct_annotation_passes(self):
        @jaxtyped(typechecker=beartype)
        def step(
            env: strux.Struct[Environment, "batch"],
        ) -> strux.Struct[Environment, "batch"]:
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
        def step(
            env: strux.Struct[Environment, "batch"],
        ) -> strux.Struct[Environment, "batch"]:
            return env

        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),  # unbatched
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        with pytest.raises(Exception):
            step(env)

    def test_functor_image_jaxtyped(self):
        @jaxtyped(typechecker=beartype)
        def same(
            a: strux.Struct[Environment, ""],
            b: strux.Struct[Environment, ""],
        ) -> strux.Struct[Environment, strux.mapped(bool)]:
            return jax.tree.map(jnp.array_equal, a, b)

        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        result = same(env, env)
        assert bool(result.walls)

    def test_nested_struct_jaxtyped(self):
        @jaxtyped(typechecker=beartype)
        def step(
            world: strux.Struct[World, "batch"],
        ) -> strux.Struct[World, "batch"]:
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
        def step(
            env: strux.Struct[LooseEnv, "batch"],
        ) -> strux.Struct[LooseEnv, "batch"]:
            return env

        # hero_pos batch=3, goal_pos batch=4 — inconsistent "batch" dim
        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
        )
        with pytest.raises(Exception):
            step(env)


# # #
# Cross-field batch consistency in the plain batched form


class TestBatchConsistency:
    def test_frankenstein_batches_rejected(self):
        # each field individually matches "b", but no single batch shape
        # is consistent across fields — the solver-backed check refuses
        import jax
        consistent = Point(x=jnp.ones(4), y=jnp.ones(4))
        _, treedef = jax.tree.flatten(consistent)
        frankenstein = jax.tree.unflatten(treedef, [jnp.ones(4), jnp.ones(5)])
        assert isinstance(consistent, strux.Struct[Point, "b"])
        assert not isinstance(frankenstein, strux.Struct[Point, "b"])

    def test_fixed_batch_tokens_checked(self):
        batched = Point(x=jnp.ones(4), y=jnp.ones(4))
        assert isinstance(batched, strux.Struct[Point, "4"])
        assert not isinstance(batched, strux.Struct[Point, "5"])

    def test_repeated_batch_names_bind_within_pattern(self):
        square = Point(x=jnp.ones((3, 3)), y=jnp.ones((3, 3)))
        oblong = Point(x=jnp.ones((3, 4)), y=jnp.ones((3, 4)))
        assert isinstance(square, strux.Struct[Point, "n n"])
        assert not isinstance(oblong, strux.Struct[Point, "n n"])

    def test_variadic_batch_pattern(self):
        batched = Point(x=jnp.ones((2, 3)), y=jnp.ones((2, 3)))
        assert isinstance(batched, strux.Struct[Point, "..."])
        assert isinstance(batched, strux.Struct[Point, "2 ..."])
        assert not isinstance(batched, strux.Struct[Point, "5 ..."])

    def test_functor_images_keep_per_field_checking(self):
        # images deliberately differ from the schema, so the solver-backed
        # consistency check does not apply to them
        import jax
        batched = Point(x=jnp.ones(4), y=jnp.ones(4))
        mask = jax.tree.map(lambda a: a > 0, batched)
        assert isinstance(mask, strux.Struct[Point, strux.mapped(bool), "b"])
