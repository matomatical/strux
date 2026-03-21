strux
=====

A JAX utility library for easily creating jit-able dataclasses.

Installation
------------

Install:

```console
pip install git+https://github.com/matomatical/strux.git
```

Dependencies: `jax`, `numpy`.

Examples
--------

Note that some readme examples use `Self` which requires python 3.11, but these
annotations are optional and otherwise strux only requires python 3.9.

### Basic usage

At the most basic level a strux struct is just a frozen dataclass registered as
a JAX pytree. It works with `jax.jit`, `jax.vmap`, `jax.tree.map`, and friends,
and supports pretty printing by default.

```python
import jax
import jax.numpy as jnp
import strux

@strux.struct
class Point:
    x: float
    y: float

p = Point(x=1.0, y=2.0)
q = Point(x=3.0, y=4.0)

# pytree operations work out of the box
r = jax.tree.map(lambda a, b: a + b, p, q)
print(r)
```

Output:
```console
Point(
  x=float(4.0),
  y=float(6.0),
)
```

### Modules with methods

Structs can hold arrays and define jit-compiled methods. Among other things,
you can use this to define neural network modules. For example, here is a
simple biased linear transformation layer module.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray  # pip install jaxtyping
from typing import Self # python3.11+
import strux

@strux.struct
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]

    @staticmethod
    @jax.jit(static_argnames=("num_inputs", "num_outputs"))
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> AffineTransform:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights=jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        biases=jnp.zeros(num_outputs)
        return AffineTransform(weights=weights, biases=biases)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights + self.biases

# initialisation
key = jax.random.key(seed=42)
net = AffineTransform.init(key=key, num_inputs=10, num_outputs=1)
print(net)

# inference
out = net.forward(jnp.ones(10))
print(out)
```

Output:
```console
AffineTransform(
  weights=jnp.float32[10,1],
  biases=jnp.float32[1],
)
[0.47424078]
```

### Submodules and static fields

Structs can be nested arbitrarily, allowing one to easily implement complex
neural networks (among other things). For example, here is a multi-layer
perceptron module that combines two of the previous AffineTransform modules.

You can use the `static_fieldnames` flag for fields that shouldn't be traced
by JAX (e.g. configuration, shapes, activation functions). These fields are
excluded from `jax.jit` and `jax.tree.map` (unlike equinox, no need for
filters). In the below example we use this to make the activation function of
the MLP configurable.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray  # pip install jaxtyping
from typing import Callable, Self  # python 3.11+
import strux

# flag 'activate' as a static field when defining the struct
@strux.struct(static_fieldnames=("activate",))
class MLP:
    linear1: AffineTransform # this is the module from the previous example ^
    linear2: AffineTransform
    activate: Callable

    @staticmethod
    @jax.jit(static_argnames=("features", "hidden", "outputs", "activate"))
    def init(
        key: PRNGKeyArray,
        features: int,
        hidden: int,
        outputs: int,
        activate: Callable = jax.nn.relu,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        return MLP(
            linear1=AffineTransform.init(k1, features, hidden),
            linear2=AffineTransform.init(k2, hidden, outputs),
            activate=activate,
        )

    @jax.jit
    def forward(self: Self, x: Float[Array, "features"]) -> Float[Array, "outputs"]:
        # because activate is static we can use it directly in jit-compiled code
        h = self.activate(self.linear1.forward(x))
        return self.linear2.forward(h)

net = MLP.init(jax.random.key(0), features=4, hidden=8, outputs=1)
print(net)
```

Output:
```console
MLP(
  linear1=AffineTransform(
    weights=jnp.float32[4,8],
    biases=jnp.float32[8],
  ),
  linear2=AffineTransform(
    weights=jnp.float32[8,1],
    biases=jnp.float32[1],
  ),
  activate=<fn:relu>,
)
```

### Vmapping and batch annotations

Structs work naturally with vectorisation and `jax.vmap`, for example for
batches of data, parameters, or anything else. You can define your struct for
the individual elements of the batch, and then annotate batched structs using
type subscripting (e.g. `Image["batch_size"]`). The result is a new struct type
with the batch dimension(s) prepended to each (non-static) field's jaxtyping
annotation.

We could use this to implement a data batch or a neural network ensemble, or
even depth-wise batches of layer parameters for use as inputs to
`jax.lax.scan`.  Here we give an example of a batched gridworld for collecting
parallel rollouts.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Bool, PRNGKeyArray  # pip install jaxtyping
from typing import Self  # python 3.11+
import strux

@strux.struct
class GridWorld:
    hero_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @staticmethod
    @jax.jit
    def init(key: PRNGKeyArray, size: int = 5) -> Self:
        walls = jax.random.bernoulli(key, 0.3, (size, size))
        hero_pos = jnp.array([0, 0])
        walls = walls.at[0, 0].set(False)
        return GridWorld(hero_pos=hero_pos, walls=walls)

    @jax.jit
    def step(self: Self, action: Int[Array, ""]) -> Self:
        deltas = jnp.array([[0,0], [-1,0], [0,-1], [1,0], [0,1]])
        new_pos = jnp.clip(self.hero_pos + deltas[action], 0, self.walls.shape[0] - 1)
        blocked = self.walls[new_pos[0], new_pos[1]]
        new_pos = jnp.where(blocked, self.hero_pos, new_pos)
        return self.replace(hero_pos=new_pos)

# initialise a batch of environments with vmap
keys = jax.random.split(jax.random.key(0), 4)
envs = jax.vmap(GridWorld.init)(keys)
print(envs)
print("hero positions before step:")
print(envs.hero_pos)

# vectorised step: up, left, down, right
actions = jnp.array([1, 2, 3, 4])
envs = jax.vmap(GridWorld.step)(envs, actions)
print("hero positions after step:")
print(envs.hero_pos)

# GridWorld["batch"] expands each field's annotation:
#   hero_pos: Int[Array, "batch 2"]
#   walls:    Bool[Array, "batch size size"]
def batched_step(
    envs: GridWorld["batch"],
    actions: Int[Array, "batch"],
) -> GridWorld["batch"]:
    return jax.vmap(GridWorld.step)(envs, actions)
```

Output:
```console
GridWorld(
  hero_pos=jnp.int32[4,2],
  walls=jnp.bool[4,5,5],
  size=int(5),
)
hero positions before step:
[[0 0]
 [0 0]
 [0 0]
 [0 0]]
hero positions after step:
[[0 0]
 [0 0]
 [1 0]
 [0 1]]
```

### Runtime type checking

Strux works together with jaxtyping's runtime type checking. For example,
if you combine it with a typechecker like beartype, shape and dtype mismatches
are caught at function boundaries.

```python
from jaxtyping import jaxtyped  # pip install jaxtyping
from beartype import beartype   # pip install beartype

@jaxtyped(typechecker=beartype)
def checked_step(
    envs: GridWorld["batch"], # GridWorld defined in previous example
    actions: Int[Array, "batch"],
) -> GridWorld["batch"]:
    return jax.vmap(GridWorld.step)(envs, actions)

# this passes: shapes and dtypes are consistent
envs = checked_step(envs, actions) # envs, actions from previous example

# this would fail: actions has wrong batch size
# checked_step(envs, jnp.array([1, 2]))  # beartype raises!
```

Development
-----------

Development has some additional optional dependencies:

```
uv pip install -e ".[dev]"
```

Installs normal dependencies plus also `jaxtyping`, `beartype`, `pytest`.

### Notes

Single-file implementation (`strux.py`, though see `tests.py` for tests).

Jaxtyping is optional for non-development installations, people should be able
to install and use strux for easily creating jit-compatible dataclasses even if
they don't use jaxtyping for type annotations. Strux detects jaxtyping
annotations via duck typing (`hasattr(hint, 'dtype')` etc.).

### Testing

Run tests with `pytest`. Make sure this passes before committing, or at least
before merging to main.

### Roadmap

Basics:

- [x] Frozen dataclass + JAX pytree registration via `@strux.struct` wrapper
- [x] Pretty printing with shape/dtype summaries for arrays
- [x] Static field support via `static_fieldnames`
- [x] Decorator syntax with keyword arguments (`@strux.struct(...)`)
- [x] Annotation-only batched type subscripting (`MyStruct["batch"]`)

Advanced features:

- [x] `isinstance` support and integrate with jaxtyping + beartype
- [ ] Pretty print registered pytree classes that aren't dataclasses
- [ ] Save/load structs to/from disk (e.g. serialisation with pytree structure)
- [ ] Support indexing and shape directly on batched structs, e.g., `env[0]`.

Project:

- [x] Test suite
- [ ] Documentation
- [ ] List on PyPI
