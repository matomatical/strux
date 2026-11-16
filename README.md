strux
=====

A JAX utility library for easily creating jit-able dataclasses.

Install:

```console
pip install git+https://github.com/matomatical/strux.git
```

Requires Python 3.12+. Depends on `jax`, `numpy`, and `jaxtyping`.

The rest of this document is an example-driven introduction to strux and its
features.

Contents:

- [Structs](#structs)
  - [Basic structs](#basic-structs)
  - [Methods and neural network modules](#methods-and-neural-network-modules)
  - [Nested structs and submodules](#nested-structs-and-submodules)
  - [Static fields](#static-fields)
  - [State and immutability](#state-and-immutability)
- [Batched structs](#batched-structs)
  - [Constructing a batch](#constructing-a-batch)
  - [Annotating batched structs](#annotating-batched-structs)
  - [Indexing and shape](#indexing-and-shape)
  - [Batch hinting](#batch-hinting)
  - [Vectorisation and network ensembles](#vectorisation-and-network-ensembles)
  - [Loop scanning and environment rollouts](#loop-scanning-and-environment-rollouts)
- [Annotation algebra](#annotation-algebra)
  - [Replacing field annotations](#replacing-field-annotations)
  - [Mapping leaf annotations](#mapping-leaf-annotations)
  - [Batching as an annotation transformation](#batching-as-an-annotation-transformation)
  - [Inserting structs at leaves](#inserting-structs-at-leaves)
  - [Composing annotations](#composing-annotations)
- [Type-checking](#type-checking)
  - [Checking an instance](#checking-an-instance)
  - [Checking a function](#checking-a-function)
  - [Static type checking](#static-type-checking)
- [Serialisation](#serialisation)
  - [Saving structs to disk](#saving-structs-to-disk)
  - [Inspecting checkpoints](#inspecting-checkpoints)
  - [Loading structs via a class template](#loading-structs-via-a-class-template)
  - [Loading structs via an instance template](#loading-structs-via-an-instance-template)
  - [Dictionary conversion and other serialisation tools](#dictionary-conversion-and-other-serialisation-tools)
  - [Checkpointing with Orbax](#checkpointing-with-orbax)
- [Other](#other)

Structs
-------

Strux's core offering is a wrapper that turns a class into a JAX-compatible
data structure called a struct. We start with a quick introduction to their
basic usage.

### Basic structs

A struct groups together multiple arrays and scalars that are parts of a single
entity in our program. We use jaxtyping annotations or Python scalars to
document the intended types of its numerical fields.

For example, here is the definition of a simple struct that represents a point
particle.

```python
import jax
from jaxtyping import Array, Float
import strux

@strux.struct
class Particle:
    position: Float[Array, "2"]
    velocity: Float[Array, "2"]
    mass: float
```

At the most basic level a strux struct is just a frozen dataclass registered as
a JAX pytree. Because it is a dataclass, we automatically get a constructor.

```python
import jax.numpy as jnp

particle = Particle(
    position=jnp.array([0.2, 0.3]),
    velocity=jnp.array([0.1, -0.2]),
    mass=1.0,
)
```

Structs support pretty printing by default, showing the types and shapes of
their array contents, and the value of scalars.

```python
print(particle)
```

```console
Particle(
  position=jnp.float32[2],
  velocity=jnp.float32[2],
  mass=float(1.0),
)
```

Because a struct is also a registered pytree, it automatically works as an
input to `jax.jit`, `jax.grad`, `jax.tree.map`, and friends. For example, we can compute the gradient of an energy functional with respect to our particle's parameters:

```python
def kinetic_energy(p: Particle) -> Float[Array, ""]:
    return 0.5 * p.mass * jnp.sum(p.velocity**2)

gradient = jax.grad(kinetic_energy)(particle)
print(gradient.position)
print(gradient.velocity)
print(gradient.mass)
```
```console
[0. 0.]
[ 0.1 -0.2]
0.025000002
```

### Methods and neural network modules

Structs can do more than just hold data. They can also define methods. And, as
registered pytrees, these methods are even compatible with JAX's function
transformations.

Among other things, this leads to an elegant basis for defining neural network
modules. For example, here is a simple biased linear transformation layer
module.

```python
# Note: jax, jnp, strux, Array, and Float were already imported above.
# In general, all examples in this README share one namespace.

from jaxtyping import PRNGKeyArray
from typing import Self

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
    ) -> "AffineTransform":
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
```

The above struct defines two methods. First, the static method `.init` is a
custom constructor that takes a PRNG key and randomly initialises a set of
weights for the network:

```python
key = jax.random.key(seed=42)
net = AffineTransform.init(key=key, num_inputs=10, num_outputs=1)
print(net)
```
```console
AffineTransform(
  weights=jnp.float32[10,1],
  biases=jnp.float32[1],
)
```

Second, `.forward` is a forward pass (or 'inference') method that takes the
struct's data as the weight and bias parameters of a neural network, applies
them to an input vector to produce the neural network's output.

```python
out = net.forward(jnp.ones(10))
print(out)
```
```console
[0.47424078]
```

Note that both methods are decorated with `jax.jit`. As far as JAX is
concerned, these methods can return structs or take structs as inputs no
problem, because structs are registered pytrees.

### Nested structs and submodules

Structs can be nested arbitrarily. Among other things, this allows one to
implement more complex neural networks. For example, here is a multi-layer
perceptron module that combines two of the previous AffineTransform modules.

```python
@strux.struct
class MLP:
    linear1: AffineTransform
    linear2: AffineTransform

    @staticmethod
    @jax.jit(static_argnames=("features", "hidden", "outputs"))
    def init(
        key: PRNGKeyArray,
        features: int,
        hidden: int,
        outputs: int,
    ) -> "MLP":
        k1, k2 = jax.random.split(key)
        return MLP(
            linear1=AffineTransform.init(k1, features, hidden),
            linear2=AffineTransform.init(k2, hidden, outputs),
        )

    @jax.jit
    def forward(self, x: Float[Array, "features"]) -> Float[Array, "outputs"]:
        y = self.linear1.forward(x)
        y = jax.nn.relu(y)
        y = self.linear2.forward(y)
        return y

# initialisation
net = MLP.init(
    key=jax.random.key(0),
    features=4,
    hidden=8,
    outputs=1,
)
print(net)

# inference
out = net.forward(jnp.ones(4))
print(out)
```
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
)
[-0.9280369]
```

The two layers have different input and output sizes. Note that the symbolic
dimension names `num_inputs` and `num_outputs` are intended to be read as being
scoped to each submodule.

### Static fields

You can use the `static_fieldnames` flag for fields that shouldn't be traced by
JAX (e.g., configuration, shapes). These fields are treated as static by
`jax.jit` and excluded from `jax.tree.map`. The effect is conceptually similar
to the use of `static_argnames` in `jax.jit`.

In the below example we use this to make the activation function of the MLP
configurable.

```python
from typing import Callable

@strux.struct(static_fieldnames=("activate",))
class ConfigurableMLP:
    linear1: AffineTransform
    linear2: AffineTransform
    activate: Callable[[Float[Array, "hidden"]], Float[Array, "hidden"]]

    @staticmethod
    @jax.jit(static_argnames=("features", "hidden", "outputs", "activate"))
    def init(
        key: PRNGKeyArray,
        features: int,
        hidden: int,
        outputs: int,
        activate: Callable[[Float[Array, "hidden"]], Float[Array, "hidden"]],
    ) -> "ConfigurableMLP":
        k1, k2 = jax.random.split(key)
        return ConfigurableMLP(
            linear1=AffineTransform.init(k1, features, hidden),
            linear2=AffineTransform.init(k2, hidden, outputs),
            activate=activate,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "features"],
    ) -> Float[Array, "outputs"]:
        y = self.linear1.forward(x)
        y = self.activate(y) # activate is static so we can use it under jit
        y = self.linear2.forward(y)
        return y

configurable_net = ConfigurableMLP.init(
    key=jax.random.key(0),
    features=4,
    hidden=8,
    outputs=1,
    activate=jnp.sin,
)
print(configurable_net)
print(configurable_net.forward(jnp.ones(4)))
```

Output:
```console
ConfigurableMLP(
  linear1=AffineTransform(
    weights=jnp.float32[4,8],
    biases=jnp.float32[8],
  ),
  linear2=AffineTransform(
    weights=jnp.float32[8,1],
    biases=jnp.float32[1],
  ),
  activate=<fn:sin>,
)
[-0.18892062]
```

### State and immutability

Strux structs are intended to be immutable. They are frozen dataclasses, which
means their fields cannot be reassigned.[^mutable] However, like with JAX
arrays, you can still write programs involving dynamically changing structs.
You can create a new version of a struct with one or more modified values by
using the struct's `.replace` method. This method is equivalent to
`dataclasses.replace`.

[^mutable]:
  Technically, a strux struct can still be mutable if its field values are
  mutable. For example, if a strux struct has a list value (still valid as a
  pytree), then that list can be modified. However, if the struct's values are
  themselves immutable, then the struct is immutable as the frozen attribute
  prevents the fields from being reassigned.


This makes it possible to write struct methods that modify a struct's values
and return the new, modified struct. One example application is defining an
environment for reinforcement learning. The environment state is represented by
a struct, and the `.step` method computes and returns a replacement struct
representing the state after the agent takes a particular action.

```python
from jaxtyping import Bool, Int

@strux.struct
class GridWorld:
    hero_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @staticmethod
    @jax.jit(static_argnames=("size",))
    def init(key: PRNGKeyArray, size: int) -> "GridWorld":
        walls = jax.random.bernoulli(key, 0.3, (size, size))
        hero_pos = jnp.array([0, 0])
        walls = walls.at[0, 0].set(False)
        return GridWorld(
            hero_pos=hero_pos,
            walls=walls,
        )

    @jax.jit
    def step(self: Self, action: int) -> Self:
        # compute the new position
        delta = jnp.array([
            [0,0],  # stay
            [-1,0], # up
            [0,-1], # left
            [1,0],  # down
            [0,1],  # right
        ])[action]
        new_pos = jnp.clip(
            self.hero_pos + delta,
            0,
            self.walls.shape[0] - 1,
        )

        # if the hero hits a wall, reset to the original position
        blocked = self.walls[new_pos[0], new_pos[1]]
        new_pos = jnp.where(blocked, self.hero_pos, new_pos)

        # create a new state with this hero position
        new_state = self.replace(hero_pos=new_pos)
        return new_state

# initialise the env
key = jax.random.key(0)
env = GridWorld.init(key=key, size=5)
print(env)
print(env.hero_pos)

# take a step
env = env.step(4)
print(env.hero_pos)
```

Output:
```console
GridWorld(
  hero_pos=jnp.int32[2],
  walls=jnp.bool[5,5],
)
[0 0]
[0 1]
```

Batched structs
---------------

Strux aims to make working with batches of data structures as ergonomic as
working with batched JAX arrays. Here we summarise features for creating and
manipulating batched structs as well as annotating functions that work with
them.

### Constructing a batch

In strux, we represent a batch of data structures as a single dataclass object
whose array leaves have a leading batch dimension.

For example, recall our `Particle` struct from above, of which `particle` was
an instance:

```python
print(particle)
```

```console
Particle(
  position=jnp.float32[2],
  velocity=jnp.float32[2],
  mass=float(1.0),
)
```

We could create a batch of three particles as follows:

```python
particles = Particle(
    position=jnp.array([
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
    ]),
    velocity=jnp.array([
        [0.1, 0.0],
        [0.0, -0.1],
        [-0.1, 0.1],
    ]),
    mass=jnp.array([
        1.0,
        2.0,
        1.0,
    ]),
)
print(particles)
```

Output:
```console
Particle(
  position=jnp.float32[3,2],
  velocity=jnp.float32[3,2],
  mass=jnp.float32[3],
)
```

### Annotating batched structs

Thus an instance of `Particle` can represent either a single particle or a
batch of particles. Conventionally, we reserve the type annotation `Particle`
to describe a single particle. We would use the type form
  `Struct[Particle, "n"]`
to describe a batch of `n` particles.

For example, the following function takes a batch of particles as input.

```python
from strux import Struct

def total_mass(
    particles: Struct[Particle, "n"],
) -> Float[Array, ""]:
    return jnp.sum(particles.mass)

print(total_mass(particles))
```

Output:
```console
4.0
```

More generally, the string follows jaxtyping's syntax to describe batched
structs with multiple batch dimensions.

Note that, so far, this is a documentation-level change---the resulting value
is still an instance of the class `Particle`, as reflected in the string output
above. However, we will see later that these annotations can also be used for
runtime type checking.

### Indexing and shape

A batched struct's *batch shape* consists of the leading dimensions beyond each
leaf array's annotated element shape.

For example, since the base `Particle` struct defines its elements to have
element shapes `(2,)`, `(2,)`, and `()` respectively, we can tell that
`particles`, with shapes `(3,2)`, `(3,2)`, and `(3,)`, has batch shape `(3,)`.

Batched structs support several convenience features that make them behave like
arrays of their batch shape:

* The `.shape` property:
  ```python
  print(particles.shape)
  ```
  ```console
  (3,)
  ```

* The `len` function:
  ```python
  print(len(particles))
  ```
  ```console
  3
  ```

* Indexing:
  ```python
  print(particles[0])
  ```
  ```console
  Particle(
    position=jnp.float32[2],
    velocity=jnp.float32[2],
    mass=jnp.float32[],
  )
  ```

* Slicing:
  ```python
  print(particles[1:].shape)
  ```
  ```console
  (2,)
  ```

* Iteration:

  ```python
  print([float(p.mass) for p in particles])
  ```
  ```console
  [1.0, 2.0, 1.0]
  ```

### Batch hinting

In many cases, strux can infer the presence of batch dimensions by comparing
leaf array dimensions to the dimensions specified in the field annotations.

However, in some situations, the original field annotations are insufficient to
distinguish batch dimensions from data dimensions.

* For example, maybe the original field annotations allow variable numbers of
  dimensions, such as a field that could hold any tensor:
    `Float[Array, "*features"]`.
  In that case, the original field annotations are ambiguous about where the
  data stops.

* As another example, maybe a reduction operation has removed some data axes,
  leaving values with shapes that no longer correspond to their original
  annotations.

If strux can't infer a batch shape, batch access (`.shape`, `len`, indexing,
slicing, iteration) will fail unless you tell strux what the batch shape is
supposed to be with the `.as_batch()` method.

For example:

```python
@strux.struct
class Ambiguous:
    data: Float[Array, "*features"]

a = Ambiguous(
    data=jnp.ones((2, 3, 4)),
)
print(a)
```
```console
Ambiguous(
  data=jnp.float32[2,3,4],
)
```

```python
# batch inference fails:
try:
    print(a[0,1])
except ValueError as e:
    print(e)
```
```console
Ambiguous: batch shape under-determined: consistent candidates {(), (2,), (2, 3), (2, 3, 4)}; annotate element dims or use .as_batch(shape=...) to choose the batch shape
```

```python
# with batch hint:
a = a.as_batch(shape=(2,3))
print(a[0,1])
```
```console
Ambiguous(
  data=jnp.float32[4],
)
```

JAX operations that construct a new struct object will clear a previous batch
hint, so you might have to re-supply the batch size information each time you
need batch access. Batch access operations themselves will preserve the
information supplied.

### Vectorisation and network ensembles

Batched structs work naturally with automatic vectorisation and `jax.vmap`.

For example, we can easily initialise a batch of neural networks by vmapping
the constructor over an array of three PRNG keys:

```python
keys = jax.random.split(jax.random.key(0), 3)
ensemble = jax.vmap(
    MLP.init,
    in_axes=(0, None, None, None),
)(
    keys,
    4, # features
    8, # hidden
    1, # output
)
```

The outer batch dimension appears on every array in the nested structure.
Compare the following two outputs.

```python
print(ensemble)
print(ensemble[0])
```
```console
MLP(
  linear1=AffineTransform(
    weights=jnp.float32[3,4,8],
    biases=jnp.float32[3,8],
  ),
  linear2=AffineTransform(
    weights=jnp.float32[3,8,1],
    biases=jnp.float32[3,1],
  ),
)
MLP(
  linear1=AffineTransform(
    weights=jnp.float32[4,8],
    biases=jnp.float32[8],
  ),
  linear2=AffineTransform(
    weights=jnp.float32[8,1],
    biases=jnp.float32[1],
  ),
)
```

As another example, we can use `vmap` again to apply each network's forward
pass to a given input:

```python
@jax.jit
def forward_ensemble(
    ensemble: Struct[MLP, "members"],
    x: Float[Array, "features"],
) -> Float[Array, "members outputs"]:
    vforward = jax.vmap(
        MLP.forward, # two-argument version of self.forward
        in_axes=(0, None),
    )
    return vforward(ensemble, x)

outputs = forward_ensemble(ensemble, jnp.ones(4))
print(outputs.shape)
```
```console
(3, 1)
```

### Loop scanning and environment rollouts

Likewise, we can use `jax.lax.scan` to iteratively construct or process the
leading batch dimension of a batched struct.

For example, we can scan an environment rollout using a state struct as a carry
and accumulate successive environment states into a trajectory batch.

```python
@jax.jit
def rollout(
    initial_state: GridWorld,
    actions: Int[Array, "time"],
) -> tuple[
    GridWorld,
    Struct[GridWorld, "time"],
]:
    def step(
        state: GridWorld,
        action: Int[Array, ""],
    ) -> tuple[GridWorld, GridWorld]:
        next_state = state.step(action)
        return next_state, next_state

    return jax.lax.scan(
        f=step,
        init=initial_state,
        xs=actions,
    )
```

We can run a trajectory of six actions from a freshly initialised state:

```python
initial_state = GridWorld.init(
    key=jax.random.key(0),
    size=5,
)
actions = jnp.array([4, 3, 4, 3, 2, 1])
final_state, trajectory = rollout(initial_state, actions)
```

The final state is a single `GridWorld`. The trajectory is a batch of six
`GridWorld` states, with a leading time dimension on each array:

```python
print(final_state)
print(trajectory)
```
```console
GridWorld(
  hero_pos=jnp.int32[2],
  walls=jnp.bool[5,5],
)
GridWorld(
  hero_pos=jnp.int32[6,2],
  walls=jnp.bool[6,5,5],
)
```

In this example, `trajectory[0]` is the state after the first action. The
initial state is not included because it was never the output of a step. Of
course, we can manually prepend the initial state in each leaf using pytree
utilities.

```python
full_trajectory = jax.tree.map(
    lambda state, states: jnp.concatenate(
        [state[None], states],
        axis=0,
    ),
    initial_state,
    trajectory,
)
print(full_trajectory)
print(full_trajectory[0].hero_pos)
```
```console
GridWorld(
  hero_pos=jnp.int32[7,2],
  walls=jnp.bool[7,5,5],
)
[0 0]
```

Annotation algebra
------------------

We have seen that the same struct class can be instantiated to hold both
singleton data and batched data. While these are the most common
instantiations, it's occasionally useful to construct more exotic combinations
of field types. Strux's `Struct` type form supports flexible annotation
expressions allowing one to denote these differences.

Note that, so far, these annotations are documentation-level changes---the
resulting values are still instances of the original class. However, we will
see later that these annotations can also be used for runtime type checking.

### Replacing field annotations

The `Replace` marker works like `.replace`, but for annotations. Named fields
receive new annotations while omitted fields keep their existing annotations.

For example, suppose we want to check which of a neural network layer's
parameters are positive. We could achieve this with a tree map:

```python
layer = net.linear1
mask = jax.tree.map(lambda x: x > 0, layer)
print(mask)
```
```console
AffineTransform(
  weights=jnp.bool[4,8],
  biases=jnp.bool[8],
)
```

But what is the appropriate annotation for this instance? By convention,
`AffineTransform` is reserved for structures with floating point data, per the
original class definition. However, this instance has Boolean arrays.

Instead, we can derive a struct annotation that modifies those field
annotations to change their type:

```python
from strux import Replace

AffineTransformMask = Struct[
    AffineTransform,
    Replace(
        weights=Bool[Array, "num_inputs num_outputs"],
        biases=Bool[Array, "num_outputs"],
    ),
]
```

We could then annotate a function that conducts the above masking operation as
follows:

```python
def positive_entries(
    layer: AffineTransform,
) -> AffineTransformMask:
    return jax.tree.map(lambda x: x > 0, layer)

mask = positive_entries(net.linear1)
print(mask)
```
```console
AffineTransform(
  weights=jnp.bool[4,8],
  biases=jnp.bool[8],
)
```

### Mapping leaf annotations

When every numerical leaf annotation undergoes the same kind of transformation,
we can use the `Map` marker.

For example, we can turn a neural network parameter pytree into a pytree that
replaces each parameter tensor with its number of elements.

```python
from strux import Map

MLPSizes = Struct[
    MLP,
    Map(lambda annotation: int),
]
```

Like `jax.tree.map`, the `Map` marker visits numerical leaf annotations through
nested structs and containers, skipping static fields. Unlike `jax.tree.map`,
the callback receives each leaf's annotation, rather than its array. The
annotations are replaced by the annotation returned by the callback. In this
case, that's the constant annotation `int`.

To compute the size tree itself, we can map the size function over the actual
arrays.

```python
def tensor_sizes(net: MLP) -> MLPSizes:
    return jax.tree.map(jnp.size, net)

sizes = tensor_sizes(net)
print(sizes)
```
```console
MLP(
  linear1=AffineTransform(
    weights=int(32),
    biases=int(8),
  ),
  linear2=AffineTransform(
    weights=int(8),
    biases=int(1),
  ),
)
```

### Batching as an annotation transformation

Prepending batch dimensions to each leaf is actually an example of `Map`.

For example, we wrote a batched neural network annotation above with a string:

```python
Ensemble1 = Struct[MLP, "members"]
```

The same annotation could also be spelled with `Map` as follows:

```python
from strux import Batch

Ensemble2 = Struct[
    MLP,
    Map(lambda annotation: Batch[annotation, "members"]),
]
```

Here, `Batch` is an annotation transformation that prepends a dimension string
to the given jaxtyping array annotation (or turns a scalar annotation into an
array annotation). For example,
  `Batch[Float[Array, "inputs outputs"], "members"]`
results in an annotation for an array of type
  `Float[Array, "members inputs outputs"]`.

`Batch` is similar to the `Shaped` annotation transformer from jaxtyping.
However, there are some subtle differences:

* `Batch` keeps separate scopes for the inner and outer annotation strings. For
  example,
    `Batch[Float[Array, "n m"], "n"]`
  does not describe the same kind of arrays as
    `Float[Array, "n n m"]`,
  but rather those of
    `Float[Array, "n_outer n_inner m_inner"]`.
  The distinction is that the two axes labelled `n` are allowed to have
  different sizes under `Batch`.

* `Batch` accepts arbitrary struct annotations, not just jaxtyping array
  annotations. For example, a third possible spelling of our ensemble type is
  ```python
  Ensemble3 = Batch[MLP, "members"]
  ```

### Inserting structs at leaves

`Map` operates on a pytree's leaf node annotations. That means the input to the
mapped function is an array or scalar annotation. The output is usually an
array or scalar annotation, but it doesn't have to be, it could also be a
struct annotation.

For example, let's say we want to compute summary statistics of the parameters
in each tensor of a neural network parameter data structure. We can create a
struct to store the mean and standard deviation of a single tensor.

```python
@strux.struct
class TensorStats:
    mean: Float[Array, ""]
    std: Float[Array, ""]
```

An appropriate structure for storing a network's summary statistics would have
the same structure as the network, but with each leaf replaced by this
`TensorStats` node with two scalar leaves.

We can annotate that structure using `Map`:

```python
MLPStats = Struct[
    MLP,
    Map(lambda annotation: TensorStats),
]
```

Then we can compute it using `jax.tree.map`:

```python
def statistics(net: MLP) -> MLPStats:
    return jax.tree.map(
        lambda array: TensorStats(
            mean=jnp.mean(array),
            std=jnp.std(array),
        ),
        net,
    )

stats = statistics(net)
print(stats)
```
```console
MLP(
  linear1=AffineTransform(
    weights=TensorStats(
      mean=jnp.float32[],
      std=jnp.float32[],
    ),
    biases=TensorStats(
      mean=jnp.float32[],
      std=jnp.float32[],
    ),
  ),
  linear2=AffineTransform(
    weights=TensorStats(
      mean=jnp.float32[],
      std=jnp.float32[],
    ),
    biases=TensorStats(
      mean=jnp.float32[],
      std=jnp.float32[],
    ),
  ),
)
```

### Composing annotations

`Struct` can take a variable number of transformations. The transformations are
applied from left to right. In general, we have the following identities:

```text
Struct[S, A, B] = Struct[Struct[S, A], B]
Struct[S, A, B, ..., Z] = Struct[Struct[...Struct[Struct[S, A], B], ...], Z]
```

For example, suppose we want to combine the `MLPSizes` and the ensemble
derivations from above, to get the sizes of the batched tensors from an
ensemble.

```python
EnsembleSizes1 = Struct[
    Ensemble1,
    Map(lambda annotation: int),
]

EnsembleSizes2 = Struct[
    Struct[
        MLP,
        "members",
    ],
    Map(lambda annotation: int),
]

EnsembleSizes3 = Struct[
    MLP,
    "members",
    Map(lambda annotation: int),
]
```

Order matters. Conversely, we could derive an annotation for a batch of size
trees:

```python
SizeEnsemble1 = Struct[
    MLPSizes,
    "members",
]

SizeEnsemble2 = Struct[
    Struct[
        MLP,
        Map(lambda annotation: int),
    ],
    "members",
]

SizeEnsemble3 = Struct[
    MLP,
    Map(lambda annotation: int),
    "members",
]
```

This feature can be used to list multiple batch axes, with consecutive batch
operations prepending new batch dimensions to the start of the list:

```python
EnsembleHistory1 = Struct[MLP, "members", "time"]
EnsembleHistory2 = Struct[MLP, "time members"]
```

Type-checking
-------------

So far, we have introduced annotations as documentation of the intended
contents of a struct. Strux also supports opt-in runtime type checking of
struct instances against annotations.

### Checking an instance

To check an instance, call `.check(annotation)`. The check compares its fields,
including nested fields, with the requirements in that annotation.

With no argument, `.check()` checks an instance's values against the class's
default field annotations, matching the annotation convention established
above.

Thus `particle.check()` checks one particle; use
`particles.check(Struct[Particle, "n"])` to check a particle batch:

```python
particle.check(Particle)
particles.check(Struct[Particle, "n"])
```

You can check a struct against any annotation derived using the annotation
algebra.

```python
mask.check(AffineTransformMask)
sizes.check(MLPSizes)
stats.check(MLPStats)
```

A successful check returns `None`, leaving the checked instance unchanged. A
mismatch raises `strux.ValidationError`:

```python
bad_particles = particles.replace(mass=jnp.ones(2))
print(bad_particles)

try:
    bad_particles.check(Struct[Particle, "n"])
except strux.ValidationError as e:
    print(e)
```
```console
Particle(
  position=jnp.float32[3,2],
  velocity=jnp.float32[3,2],
  mass=jnp.float32[2],
)
Particle.mass: batch dimension 'n' has size 2; expected 3.
  Expected shape: (n,)
  Actual shape:   (2,)
```

### Checking a function

Strux also supports runtime typechecking of functions, so that shape and dtype
mismatches can be caught at function boundaries. A function wrapped with
`@strux.checked` validates arguments and return values against their
annotations.

This feature builds upon and extends jaxtyping's runtime type checking
functionality, using the
  [beartype](https://beartype.readthedocs.io/en/latest/)
backend. As a result, it must be enabled by installing strux with the optional
dependency
  `strux[beartype]`.

For example, here is a function that applies each network in an ensemble to its
own input vector.

```python
@strux.checked
def checked_forward(
    ensemble: Struct[MLP, "members"],
    inputs: Float[Array, "members features"],
) -> Float[Array, "members outputs"]:
    vforward = jax.vmap(
        MLP.forward,
        in_axes=(0, 0),
    )
    return vforward(ensemble, inputs)
```

Note that the annotations share a scope for top-level axis names. Since we use
the same string `"members"` across the three annotations, the number of
networks, input vectors, and output vectors need to match. Our ensemble has
three networks, so we supply three input vectors:

```python
inputs = jnp.ones((3, 4))
outputs = checked_forward(ensemble, inputs)
print(outputs.shape)
```
```console
(3, 1)
```

On the other hand, a call with two inputs and three networks fails:

```python
from jaxtyping import TypeCheckError

try:
    checked_forward(ensemble, jnp.ones((2, 4)))
except TypeCheckError as e:
    print(e)
```
```console
Type-check error whilst checking the parameters of __main__.checked_forward.
The problem arose whilst typechecking parameter 'inputs'.
Actual value: f32[2,4](jax)
Expected type: <class 'Float[Array, 'members features']'>.
----------------------
Called with parameters: {
  'ensemble':
  MLP(
    linear1=AffineTransform(weights=f32[3,4,8](jax), biases=f32[3,8](jax)),
    linear2=AffineTransform(weights=f32[3,8,1](jax), biases=f32[3,1](jax))
  ),
  'inputs': f32[2,4](jax)
}
Parameter annotations: (ensemble: strux.annotate.Struct[MLP, 'members'], inputs: Float[Array, 'members features']) -> Any.
The current values for each jaxtyping axis annotation are as follows.
members=3
```

Function-boundary checking supports any struct annotation including all of the
transformed annotations described in the annotation algebra section. Since it
builds on jaxtyping's runtime type checking, it also checks plain jaxtyping
array annotations.

When `jax.jit` wraps a function, checks on internal function boundaries occur
at trace time only, rather than on every invocation.

### Static type checking

Static checkers such as mypy see `Struct[Particle, "n"]` as `Particle`, the
class. They can check ordinary declared field types, but do not understand the
transformations described by strux annotations, or the shapes and dtypes
described by jaxtyping annotations.

Serialisation
-------------

Strux provides methods for saving and restoring structs, as well as functions
for converting them to dictionaries and inspecting saved checkpoints.

### Saving structs to disk

Structs can be saved to disk to be restored later, using the `save` method.
This creates a file that stores most of the information needed to later
restore a struct:

* The values of any array or scalar leaves.
* Static information about the array and scalar leaves, including their shapes
  and dtypes.
* Values of static fields that can be easily serialised (literal values for
  static fields involving Python builtins or collections).
* Some static information about pytree structure that is not specified in the
  defining class annotations, such as tags for union types.

The remaining pytree structure and Python values that are hard to serialise
(such as functions) are not saved.

For example, the following snippet saves the `ConfigurableMLP` instance with
the sine activation defined earlier. We will see that the MLP's weight and bias
arrays are serialised, but the activation function cannot be.

```python
import os
import tempfile

path = os.path.join(
    tempfile.mkdtemp(),
    "configurable_mlp.npz",
)
configurable_net.save(path)
```

Two file formats are supported:

* If the filename has a `.npz` extension then the data fields are saved in
  compressed numpy format (pass `fmt="savez"` for uncompressed).

* If the filename has a `.safetensors` extension, and strux is installed with
  the optional `safetensors` dependency (`pip install strux[safetensors]`),
  then strux uses the [safetensors](https://huggingface.co/docs/safetensors/)
  memory-mapped format.

Writes are atomic. By default, saving refuses to overwrite an existing file;
pass `overwrite=True` to replace it (e.g. for repeatedly saving the latest
checkpoint during training).

### Inspecting checkpoints

To see what was saved, use `strux.describe`:

```python
print(strux.describe(path))
```

This function inspects a saved file and renders its recorded leaf data and
structural information without constructing the actual structs.

The header shows the path to the saved file:
<!--output:illustrative-->
```console
configurable_mlp.npz (savez_compressed, strux format 2): 4 arrays, 49 elements, 196 B
__main__.ConfigurableMLP
  linear1: __main__.AffineTransform
    weights: float32[4 8]
    biases: float32[8]
  linear2: __main__.AffineTransform
    weights: float32[8 1]
    biases: float32[1]
```

The same functionality is available via the command line:

```console
$ python -m strux configurable_mlp.npz
configurable_mlp.npz (savez_compressed, strux format 2): 4 arrays, 49 elements, 196 B
__main__.ConfigurableMLP
  linear1: __main__.AffineTransform
    weights: float32[4 8]
    biases: float32[8]
  linear2: __main__.AffineTransform
    weights: float32[8 1]
    biases: float32[1]
```

### Loading structs via a class template

There are two ways to restore structs saved to disk. The first method is to use
the module-level `strux.load` function with a struct class as a template.

In this case, a struct class's annotations provide the overarching pytree
structure. The concrete shapes and dtypes of arrays as well as their values are
taken from the file, as are the values of static fields that were serialised.

This might not cover all static information. If there are non-serialisable
static fields, their values can be provided via `statics=`, a dict keyed by
`/`-separated field name paths. If a static field's value is still missing and
the class gives it a default value, then that default value will be used.
Otherwise, an error will be raised.

For example, to restore our configurable MLP, we need to supply the activation
function in a static values dictionary.

```python
restored = strux.load(
    path,
    template=ConfigurableMLP,
    statics={"activate": jnp.sin},
)
```

We can verify that the restored parameters match those we saved:

```python
equal_leaves = jax.tree.map(
    jnp.array_equal,
    configurable_net,
    restored,
)
print(jax.tree.all(equal_leaves))
```
```console
True
```

By default the format of the file is inferred from the file extension. The
optional `fmt=` keyword on `strux.load` allows one to override this to
`"savez"` for numpy (compressed or uncompressed) or `"safetensors"` for
safetensors.

### Loading structs via an instance template

The second method for restoring a struct saved to disk is to use the
module-level `strux.load` function with a struct *instance* as a template, or,
equivalently, to use the `.restore()` method of a template instance.

In this case, a template instance supplies the concrete pytree structure
including all static information. However, for static information that is
recorded in the file, including shapes and dtypes of array leaves, strux raises
an error in the case of a mismatch with the template. Note that some static
information, such as the value of function fields, cannot be checked like this
as it is only available from the template, not from the file.

For example, in order to restore our configurable MLP using this method, we can
instantiate a new MLP with the same architecture and activation function but
random weights. Then, we can restore the saved weights into a new instance.

```python
template = ConfigurableMLP.init(
    key=jax.random.key(999),
    features=4,
    hidden=8,
    outputs=1,
    activate=jnp.sin,
)
restored = template.restore(path)
# equivalently: strux.load(path, template=template)
```

Once again, we can verify that the restored parameters match the saved model:

```python
equal_leaves = jax.tree.map(
    jnp.array_equal,
    configurable_net,
    restored,
)
print(jax.tree.all(equal_leaves))
```
```console
True
```

By default the format of the file is inferred from the file extension. The
optional `fmt=` keyword on `.restore` allows one to override this to `"savez"`
for numpy (compressed or uncompressed) or `"safetensors"` for safetensors.


### Dictionary conversion and other serialisation tools

You can also use `strux.to_dict` and `strux.from_dict` to convert structs to
and from flat dictionaries of arrays.

For example, the keys in our network's dictionary identify its parameter
fields:

```python
arrays = strux.to_dict(configurable_net)
print(sorted(arrays))
```
```console
['linear1/biases', 'linear1/weights', 'linear2/biases', 'linear2/weights']
```

An instance template supplies the structure and static values when converting
back to a struct:

```python
restored = strux.from_dict(
    arrays,
    template=configurable_net,
)
print(type(restored) is ConfigurableMLP)
```
```console
True
```

These dictionaries don't contain any static information other than the paths,
shapes, and dtypes of the arrays. The structural metadata that `save` would
record alongside this data can be produced by the function
  `strux.metadata`.
Passing this object back to `from_dict` as `meta=` when restoring from a class
template will use it to repopulate the struct with the appropriate static
information. This restoration then follows the same rules as loading from disk:
recorded static values are restored with a class template and checked against
an instance template. Unrecorded values, such as activation functions, come
from the instance template, or from `statics=` and class defaults when using a
class template.

These functions can compose with other Python serialisation tools.


### Checkpointing with Orbax

If you prefer to process your checkpoints with
  [Orbax](https://orbax.readthedocs.io/),
you'll be pleased to know that, as strux structs are registered pytrees, they
can be handled natively by Orbax.

For example, using the same network:

```python
import orbax.checkpoint as ocp  # pip install orbax-checkpoint

orbax_path = os.path.join(
    tempfile.mkdtemp(),
    "configurable_mlp_orbax",
)
with ocp.StandardCheckpointer() as checkpointer:
    checkpointer.save(orbax_path, configurable_net)
    checkpointer.wait_until_finished()
```

Saving is asynchronous, so we wait for it to finish before reading the
checkpoint. We can then restore it using a template:

```python
with ocp.StandardCheckpointer() as checkpointer:
    restored = checkpointer.restore(
        orbax_path,
        target=template,
    )

equal_leaves = jax.tree.map(
    jnp.array_equal,
    configurable_net,
    restored,
)
print(jax.tree.all(equal_leaves))
```
```console
True
```

This approach uses Orbax's policies regarding saving and restoring static
information. The details are out of scope for this tutorial, see Orbax's
documentation.

Other
-----

Features we implemented already but I haven't gotten around to documenting
properly:

* The `InstanceOf[S]` annotation if you want `.check` or `@strux.checked` to
  check only `isinstance(value, S)`, without checking the fields.
* `Map` callbacks that change a leaf's dtype while preserving its shape, or
  change its shape while preserving its dtype.
* Struct fields containing lists, tuples, dictionaries, optional values, or
  batches of other structs.
* Checking methods with `Self` annotations, choosing a backend with
  `@strux.checked(typechecker=...)`, and disabling automatic checks with
  `typechecker=None`.
* The `.size` property for counting all numerical elements in a struct, and
  `strux.tree_dims` for reading the sizes of its declared named dimensions.
* Controlling pretty-printing depth and indentation with `strux.to_str`, or
  with format strings such as `f"{net:2.4}"`.
* Module-level equivalents of the convenience methods and properties, such as
  `strux.check`, `strux.as_batch`, and `strux.tree_shape`.

Possible future additions:

* Dedicated shape and dtype transformation helpers.
* An import hook for enabling `strux.checked` across a codebase, like in
  jaxtyping.
