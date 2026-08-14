"""
Tests for the batch-shape solver: batch-tolerant constructor checking and
schema-driven construction and batch inference.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype

import strux

from example_structs import Environment


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
        # a python scalar is one element's worth of data (batch () only);
        # batching a scalar field means holding an array (strux never
        # broadcasts a scalar across a batch)
        @jaxtyped(typechecker=beartype)
        @strux.struct
        class Config:
            coeff: float
            u: Float[Array, ""]

        Config(coeff=0.5, u=jnp.float32(1.0))
        Config(coeff=jnp.ones(3), u=jnp.ones(3))
        with pytest.raises(Exception):
            Config(coeff=0.5, u=jnp.ones(3))
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
        assert isinstance(b, strux.Struct[Scaled, "batch"])
        assert not isinstance(s, strux.Struct[Scaled, "batch"])
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
        assert isinstance(batched, strux.Struct[Bank, "batch"])
        assert not isinstance(bank, strux.Struct[Bank, "batch"])
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
        assert isinstance(batched, strux.Struct[Sum, "batch"])
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
        assert isinstance(Holder(env=db), strux.Struct[Holder, "batch"])

    def test_python_scalars_constrain_batch_to_unbatched(self):
        @strux.struct
        class Config:
            coeff: float
            u: Float[Array, ""]

        assert Config(coeff=0.5, u=jnp.float32(1.0)).shape == ()
        # a batched struct holds arrays in its scalar-annotated fields
        assert Config(coeff=jnp.ones(3), u=jnp.ones(3)).shape == (3,)
        with pytest.raises(strux.ValidationError):
            Config(coeff=0.5, u=jnp.ones(3))
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
