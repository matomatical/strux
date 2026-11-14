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
        # an explicit check
        class Impostor:
            dtype = None
            array_type = None
            dim_str = "2"

        @strux.struct
        class HasImpostor:
            pos: Impostor

        with pytest.raises(strux.SchemaError, match="annotated dataclass"):
            HasImpostor(pos=Impostor()).check(strux.Struct[HasImpostor, "..."])

    def test_plain_scalar_hint_wrong_dtype_fails(self):
        @strux.struct
        class Loss:
            loss: float

        batched = Loss(loss=jnp.zeros(4, dtype=jnp.int32))
        assert not isinstance(batched, strux.Struct[Loss, "batch"])

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


# # #
# Functor images


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
        @strux.struct
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
        @strux.struct
        class LooseChild:
            pos: Int[Array, "2"]

        @strux.struct
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
        ) -> strux.Struct[Environment, strux.Map(lambda _: bool)]:
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
        @strux.struct
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
        assert isinstance(mask, strux.Struct[Point, strux.Map(lambda _: bool), "b"])


# # #
# Stacked fields under the form


class TestStackedFieldImages:
    """
    A `Struct[Cls, dims]` *field* declares dims that sit inside whatever
    batch the enclosing struct carries, so the form applied to the
    enclosing struct prepends its own dims in front of the declared ones.
    """

    def _classes(self):
        @strux.struct
        class Block:
            scale: Float[Array, "embed"]

        @strux.struct
        class Network:
            embedding: Float[Array, "embed"]
            blocks: strux.Struct[Block, "depth"]

        return Block, Network

    def _network(self, Block, Network, batch=()):
        return Network(
            embedding=jnp.ones((*batch, 4)),
            blocks=Block(scale=jnp.ones((*batch, 3, 4))),
        )

    def test_rank_exact_form_accepts_the_unbatched_network(self):
        Block, Network = self._classes()
        network = self._network(Block, Network)
        assert isinstance(network, strux.Struct[Network, ""])

    def test_form_accepts_the_batched_image(self):
        Block, Network = self._classes()
        batched = self._network(Block, Network, batch=(2,))
        assert isinstance(batched, strux.Struct[Network, "members"])
        assert not isinstance(batched, strux.Struct[Network, ""])

    def test_form_refuses_the_wrong_number_of_prepended_dims(self):
        Block, Network = self._classes()
        batched = self._network(Block, Network, batch=(2,))
        assert not isinstance(batched, strux.Struct[Network, "a b"])

    def test_declared_dims_missing_from_the_value_are_refused(self):
        # the declared dims are a promise about the field's leaves, so a
        # network whose blocks lack them fails explicit checking
        Block, Network = self._classes()
        with pytest.raises(strux.ValidationError, match="Network.blocks.scale"):
            Network(embedding=jnp.ones(4), blocks=Block(scale=jnp.ones(4))).check(
                strux.Struct[Network, "..."]
            )
