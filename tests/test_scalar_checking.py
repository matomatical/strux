"""Scalar values retain their representations across checked JAX boundaries."""

import inspect
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from jaxtyping import Complex64, Float16, Float32, Float64, Int8, TypeCheckError

import strux


@pytest.mark.parametrize(
    "scalar,value,wrong_array",
    [
        (bool, True, jnp.array(1)),
        (int, 2, jnp.array(1.5)),
        (float, 1.5, jnp.array(1)),
        (complex, 1 + 2j, jnp.array(1.5)),
    ],
)
def test_scalar_fields_and_boundaries_agree(scalar, value, wrong_array):
    @strux.struct
    class Holder:
        item: scalar

    @strux.checked
    def identity(item: scalar) -> scalar:
        return item

    assert identity.__annotations__["item"] is scalar
    assert inspect.signature(identity).return_annotation is scalar
    for item in (value, np.asarray(value), np.asarray(value)[()], jnp.asarray(value)):
        Holder(item).check()
        assert identity(item) is item
        assert jnp.array_equal(jax.jit(identity)(item), jnp.asarray(value))

    batch = jnp.asarray([value, value])
    assert jnp.array_equal(jax.vmap(identity)(batch), batch)
    wrong_numpy_array = np.asarray(wrong_array)
    for item in (
        "wrong",
        batch,
        wrong_array,
        wrong_numpy_array,
        wrong_numpy_array[()],
        wrong_numpy_array.item(),
    ):
        with pytest.raises(strux.ValidationError):
            Holder(item).check()
        with pytest.raises(TypeCheckError, match="parameters"):
            identity(item)
    with pytest.raises(TypeCheckError, match="parameters"):
        jax.jit(identity)(wrong_array)

    @strux.checked
    def wrong_return() -> scalar:
        return batch

    with pytest.raises(TypeCheckError, match="return value"):
        wrong_return()


def test_scalar_aliases_containers_and_metadata():
    type Count = int
    type Counts[T] = tuple[T, list[T]]

    @strux.checked
    def identity(value: Counts[Count]) -> Counts[Count]:
        return value

    value = (jnp.array(1), [jnp.array(2)])
    assert identity(value) is value
    with pytest.raises(TypeCheckError):
        identity((jnp.ones(2, dtype=int), [jnp.array(2)]))
    with pytest.raises(TypeCheckError):
        identity((jnp.array(1), [jnp.array(2.5)]))

    @strux.checked
    def annotated(value: Annotated[float, "scalar"]) -> float:
        return value

    assert annotated(jnp.array(1.5)) == 1.5
    with pytest.raises(TypeCheckError):
        annotated(jnp.ones(2))


@pytest.mark.parametrize(
    "scalar,python_value", [(bool, True), (int, 1), (float, 1.5), (complex, 1 + 2j)]
)
def test_optional_scalar_fields_and_boundaries_agree(scalar, python_value):
    @strux.struct
    class OptionalScalar:
        value: scalar | None

    @strux.checked
    def optional(value: scalar | None) -> scalar | None:
        return value

    for value in (
        None,
        python_value,
        jnp.asarray(python_value),
        np.asarray(python_value),
        np.asarray(python_value)[()],
    ):
        OptionalScalar(value).check()
        assert optional(value) is value
        traced = jax.jit(optional)(value)
        if value is None:
            assert traced is None
        else:
            assert jnp.array_equal(traced, jnp.asarray(value))

    value = jnp.asarray([python_value])
    with pytest.raises(strux.ValidationError):
        OptionalScalar(value).check()
    with pytest.raises(TypeCheckError):
        optional(value)


def test_optional_container_preserves_scalar_convention():
    @strux.struct
    class OptionalList:
        value: list[int] | None

    @strux.checked
    def optional_list(value: list[int] | None) -> list[int] | None:
        return value

    values = [jnp.array(1)]
    OptionalList(values).check()
    assert optional_list(values) is values


@pytest.mark.parametrize("value", [None, 1.5, np.float32(1.5), jnp.array(1.5)])
def test_optional_scalar_checkpoint_roundtrip(tmp_path, value):
    @strux.struct
    class Holder:
        value: float | None

    path = tmp_path / "optional.npz"
    Holder(value).save(path)
    restored = strux.load(path, template=Holder)
    restored.check()
    if value is None:
        assert restored.value is None
    else:
        assert restored.value == value


@pytest.mark.parametrize(
    "dtype,exact_value,other_value",
    [
        (Float16, np.float16(1.5), np.float32(1.5)),
        (Float32, np.float32(1.5), np.float64(1.5)),
        (Float64, np.float64(1.5), np.float32(1.5)),
        (Int8, np.int8(1), np.int32(1)),
        (Complex64, np.complex64(1j), np.complex128(1j)),
    ],
)
def test_concrete_array_dtypes_are_strict_but_arraylike_scalar_arms_expand(
    dtype, exact_value, other_value
):
    exact = dtype[jax.Array | np.ndarray, ""]
    arraylike = dtype[ArrayLike, ""]

    @strux.struct
    class Exact:
        value: exact

    @strux.struct
    class Arraylike:
        value: arraylike

    @strux.checked
    def precise(value: exact) -> exact:
        return value

    @strux.checked
    def broad(value: arraylike) -> arraylike:
        return value

    value = np.asarray(exact_value)
    Exact(value).check()
    assert precise(value) is value

    for value in (np.asarray(other_value), exact_value.item()):
        with pytest.raises(strux.ValidationError):
            Exact(value).check()
        with pytest.raises(TypeCheckError):
            precise(value)
        Arraylike(value).check()
        assert broad(value) is value

    @strux.checked
    def wrong_return() -> exact:
        return np.asarray(other_value)

    with pytest.raises(TypeCheckError, match="return value"):
        wrong_return()


def test_union_members_do_not_narrow_scalar_shorthand():
    annotation = Float32[jax.Array, ""] | float | None

    @strux.struct
    class Holder:
        value: annotation

    @strux.checked
    def identity(value: annotation) -> annotation:
        return value

    for value in (None, 1.5, np.float16(1.5), np.array(1.5, dtype=np.float64)):
        Holder(value).check()
        assert identity(value) is value
    for value in (1, np.int64(1), np.array([1.5])):
        with pytest.raises(strux.ValidationError):
            Holder(value).check()
        with pytest.raises(TypeCheckError):
            identity(value)


def test_scalar_class_positions_and_explicit_instance_checks_remain_nominal():
    class Handler[T: int]:
        pass

    @strux.checked
    def handler(value: Handler[int]) -> Handler[int]:
        return value

    value = Handler()
    assert handler(value) is value
    with pytest.raises(TypeCheckError):
        handler(42)

    @strux.checked
    def class_object(value: type[int]) -> type[int]:
        return value

    assert class_object(int) is int
    with pytest.raises(TypeCheckError):
        class_object(jnp.int32)

    @strux.checked
    def python_instance(value: strux.InstanceOf[int]) -> strux.InstanceOf[int]:
        return value

    assert python_instance(1) == 1
    with pytest.raises(TypeCheckError):
        python_instance(jnp.array(1))

    @strux.checked
    def literal(value: Literal[1]) -> Literal[1]:
        return value

    assert literal(1) == 1
    with pytest.raises(TypeCheckError):
        literal(2)
