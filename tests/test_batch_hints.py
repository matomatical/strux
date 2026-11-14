"""Explicit batch views share data and cooperate with native JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

import strux


@strux.struct
class Layer:
    weights: Float[Array, "inputs outputs"]
    bias: Float[Array, "outputs"]


@strux.struct
class Model:
    layer: Layer


def summaries():
    return Model(Layer(jnp.arange(24.0).reshape(3, 4, 2), jnp.ones((3, 4, 2))))


def test_view_preserves_classes_data_treedef_and_original():
    value = summaries()
    view = value.as_batch(shape=(3, 4))
    assert type(view) is Model and type(view.layer) is Layer
    assert view is not value
    assert view.layer.weights is value.layer.weights
    assert jax.tree.structure(view) == jax.tree.structure(value)
    assert "_strux_batch_shape" not in vars(value)
    assert "_strux_batch_shape" not in vars(value.layer)
    assert view.shape == view.layer.shape == (3, 4)


@pytest.mark.parametrize(
    "index, shape",
    [
        (0, (4,)),
        ((slice(None), 1), (3,)),
        ((Ellipsis, 1), (3,)),
        ((None, Ellipsis), (1, 3, 4)),
        ((slice(None, None, -1), slice(1, 3)), (3, 2)),
        ((1, 2), ()),
        ((np.int64(1),), (4,)),
    ],
)
def test_index_consumes_batch_axes_only(index, shape):
    view = summaries().as_batch(shape=(3, 4))
    result = view[index]
    assert result.shape == result.layer.shape == shape
    assert result.layer.weights.shape == (*shape, 2)


def test_too_many_indices_and_bounds():
    view = summaries().as_batch(shape=(3, 4))
    with pytest.raises(IndexError):
        view[:, :, 0]
    with pytest.raises(IndexError):
        view[3]
    with pytest.raises(IndexError):
        view[..., 4]
    with pytest.raises(TypeError):
        view[0, 0][0]


def test_jit_dynamic_index_and_map_drop_hints():
    value = summaries()

    @jax.jit
    def select(value, index):
        result = value.as_batch(shape=(3, 4))[index]
        assert result.shape == (4,)
        return result

    result = select(value, jnp.int32(1))
    np.testing.assert_array_equal(result.layer.weights, value.layer.weights[1])
    assert "_strux_batch_shape" not in vars(result)
    view = value.as_batch(shape=(3, 4))
    assert "_strux_batch_shape" not in vars(jax.tree.map(lambda x: x, view))
    assert "_strux_batch_shape" not in vars(view.replace())


def test_integer_array_index_under_jit():
    @jax.jit
    def take(value, indices):
        return value.as_batch(shape=(3, 4))[indices].layer.weights

    value = summaries()
    np.testing.assert_array_equal(
        take(value, jnp.array([2, 0])), value.layer.weights[jnp.array([2, 0])]
    )


def test_hint_prefix_and_sizes_are_checked():
    value = summaries()
    for shape in ((4, 3), (3, 4, 3), (-1,), (True,)):
        with pytest.raises((TypeError, ValueError)):
            value.as_batch(shape=shape)
    with pytest.raises(TypeError):
        value.as_batch(shape=(3.0,))


def test_dtype_changes_do_not_disable_default_batch_access():
    layer = Layer(jnp.ones((3, 2, 4)), jnp.ones((3, 4)))
    mask = jax.tree.map(lambda x: x > 0, layer)
    assert mask.shape == (3,)
    assert mask[0].weights.shape == (2, 4)
    with pytest.raises(strux.ValidationError):
        mask.check(strux.Struct[Layer, "n"])


def test_mutable_container_changes_are_not_hidden_by_shape_caches():
    @strux.struct
    class Collection:
        arrays: list[Float[Array, "2"]]

    value = Collection([jnp.ones((3, 2))])
    assert value.shape == (3,)
    value.arrays[0] = jnp.ones((5, 2))
    assert value.shape == (5,)


def test_python_scalar_type_changes_do_not_disable_shape_queries():
    @strux.struct
    class Scalar:
        value: float

    value = Scalar(1.0)
    mask = jax.tree.map(lambda x: x > 0, value)
    assert mask.shape == ()
    with pytest.raises(strux.ValidationError):
        mask.check()


def test_boolean_indices_consume_only_their_batch_axes():
    value = summaries().as_batch(shape=(3, 4))
    mask = jnp.arange(12).reshape(3, 4) % 2 == 0
    selected = value[mask]
    assert selected.shape == (6,)
    np.testing.assert_array_equal(selected.layer.weights, value.layer.weights[mask])
    assert value[True, :, :].shape == (1, 3, 4)
    assert value[False, :, :].shape == (0, 3, 4)
    with pytest.raises(IndexError, match="too many"):
        value[jnp.ones((3, 4, 2), bool)]
