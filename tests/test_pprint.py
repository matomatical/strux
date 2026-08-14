"""
Tests for pretty printing (to_str).
"""

import jax
import jax.numpy as jnp

import strux

from example_structs import Environment, World


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

    # custom pytree classes (not dataclasses) render via the registry
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
        result = strux.to_str(MyNode(1.0, jnp.zeros(3)))
        assert result.startswith("MyNode(")
        assert "float(1.0)," in result
        assert "jnp.float32[3]," in result

    def test_custom_pytree_class_with_keys(self):
        class KeyedNode:
            def __init__(self, a):
                self.a = a

        jax.tree_util.register_pytree_with_keys(
            KeyedNode,
            lambda n: ([(jax.tree_util.GetAttrKey("a"), n.a)], None),
            lambda _, children: KeyedNode(*children),
        )
        result = strux.to_str(KeyedNode(jnp.ones(2)))
        assert result == "KeyedNode(\n  a=jnp.float32[2],\n)"

    def test_custom_pytree_class_max_depth(self):
        class DeepNode:
            def __init__(self, a):
                self.a = a

        jax.tree_util.register_pytree_node(
            DeepNode,
            lambda n: ((n.a,), None),
            lambda _, children: DeepNode(*children),
        )
        assert strux.to_str(DeepNode(1.0), max_depth=0) == "DeepNode(...)"


def test_render_shows_declared_fields_only():
    # the instance __dict__ carries cached solver state (set at checked
    # construction, or lazily by .shape); rendering must not show it
    import strux
    from jaxtyping import Array, Float

    @strux.struct
    class Pair:
        x: Float[Array, ""]
        y: Float[Array, ""]

    p = Pair(x=jnp.ones(3), y=jnp.zeros(3))
    assert p.shape == (3,)      # populates the cache on this instance
    assert str(p) == "Pair(\n  x=jnp.float32[3],\n  y=jnp.float32[3],\n)"
