strux
=====

A JAX utility library for easily creating jit-able dataclasses.

> **Draft for strux 0.5.** Prepared by Codex for Matthew's review, using
> passages and examples from the existing README. This describes the proposed
> API, including features that are not implemented yet. In particular,
> unchecked construction and replacement, the new annotation algebra,
> `.as_batch(...)`, and `.check(...)` are targets for the rebuild. The
> examples are written to run in order once that API exists. Outputs for
> unimplemented features show intended results, not passing tests.

Contents
--------

- [Installation](#installation)
- [Structs](#structs)
  - [Basic usage](#basic-usage)
  - [Modules with methods](#modules-with-methods)
  - [Submodules](#submodules)
  - [Static fields](#static-fields)
  - [Environment states and replacement](#environment-states-and-replacement)
- [Batched structs](#batched-structs)
  - [Constructing a batch](#constructing-a-batch)
  - [Indexing and shape](#indexing-and-shape)
  - [Network ensembles](#network-ensembles)
  - [Parallel rollouts](#parallel-rollouts)
- [Annotation algebra](#annotation-algebra)
  - [Replacing field annotations](#replacing-field-annotations)
  - [Mapping leaf annotations](#mapping-leaf-annotations)
  - [Batching as an annotation transformation](#batching-as-an-annotation-transformation)
  - [Inserting a struct at a leaf](#inserting-a-struct-at-a-leaf)
  - [Composing annotations](#composing-annotations)
  - [Batch hints](#batch-hints)
- [Type-checking](#type-checking)
  - [Checking a value](#checking-a-value)
  - [Checking function boundaries](#checking-function-boundaries)
- [Serialisation](#serialisation)
  - [Saving structs to disk](#saving-structs-to-disk)
  - [Loading structs from disk](#loading-structs-from-disk)
  - [Inspecting checkpoints](#inspecting-checkpoints)
  - [Other serialisation tools](#other-serialisation-tools)
  - [Orbax](#orbax)

Installation
------------

Install:

```console
pip install git+https://github.com/matomatical/strux.git
```

Requires python 3.12+. Dependencies: `jax`, `numpy`, `jaxtyping`.

The installation command above installs the current library. The proposed
features in this draft are being developed on the `rebuild-0.5` branch.

Structs
-------

### Basic usage

At the most basic level a strux struct is just a frozen dataclass registered as
a JAX pytree. It works with `jax.jit`, `jax.vmap`, `jax.tree.map`, and friends,
and supports pretty printing by default.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import strux

@strux.struct
class Particle:
    position: Float[Array, "2"]
    velocity: Float[Array, "2"]
    mass: Float[Array, ""]

particle = Particle(
    position=jnp.array([0.2, 0.3]),
    velocity=jnp.array([0.1, -0.2]),
    mass=jnp.array(1.0),
)
print(particle)
```

Output:
```console
Particle(
  position=jnp.float32[2],
  velocity=jnp.float32[2],
  mass=jnp.float32[],
)
```

The field annotations describe one particle: two vectors and a scalar.
Numerical array fields use jaxtyping annotations specifying the dtype and
shape. An empty shape string means a scalar array. Dimension names can also
be symbolic, as in the next example.

Pytree operations work out of the box:

```python
doubled = jax.tree.map(lambda x: 2 * x, particle)
print(doubled.mass)
```

Output:
```console
2.0
```

Construction assigns the fields without checking their shapes or dtypes.
Annotations describe the intended contents; checking them is optional.

### Modules with methods

Structs can hold arrays and define jit-compiled methods. Among other things,
you can use this to define neural network modules. For example, here is a
simple biased linear transformation layer module.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Self
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
    ) -> Self:
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

### Submodules

Structs can be nested arbitrarily, allowing one to easily implement complex
neural networks (among other things). For example, here is a multi-layer
perceptron module that combines two of the previous AffineTransform modules.

```python
@strux.struct
class MLP:
    linear1: AffineTransform
    linear2: AffineTransform

    @jax.jit
    def forward(self, x: Float[Array, "features"]) -> Float[Array, "outputs"]:
        h = jax.nn.relu(self.linear1.forward(x))
        return self.linear2.forward(h)

k1, k2 = jax.random.split(jax.random.key(0))
net = MLP(
    linear1=AffineTransform.init(k1, num_inputs=4, num_outputs=8),
    linear2=AffineTransform.init(k2, num_inputs=8, num_outputs=1),
)
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
)
```

The two layers have different input and output sizes. Dimension names such
as `num_inputs` and `num_outputs` describe each layer independently.

### Static fields

You can use the `static_fieldnames` flag for fields that shouldn't be traced
by JAX (e.g. configuration, shapes). These fields are treated as static by
`jax.jit` and excluded from `jax.tree.map`. In the below example we use this
to make the activation function of the MLP configurable.

Here is a replacement definition of `MLP`, also adding an initialiser. The
remaining examples use this version.

```python
from typing import Callable

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

Unlike in equinox, the decision of whether a field is static happens at class
definition time. This removes the need for filters and filtered JAX
transformations at a slight flexibility cost.

### Environment states and replacement

Structs are also useful for environment states. Here is a gridworld with
walls and a moving agent. The `step` method returns a new state, leaving the
original unchanged.

```python
from jaxtyping import Bool, Int

@strux.struct
class GridWorld:
    hero_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @staticmethod
    @jax.jit(static_argnames=("size",))
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

env = GridWorld(
    hero_pos=jnp.array([0, 0]),
    walls=jnp.zeros((5, 5), dtype=bool),
)
next_env = env.step(jnp.array(4))
print(next_env)
print(env.hero_pos, next_env.hero_pos)
```

Output:
```console
GridWorld(
  hero_pos=jnp.int32[2],
  walls=jnp.bool[5,5],
)
[0 0] [0 1]
```

`env.replace(hero_pos=new_pos)` returns a new instance with that field
changed. The other fields keep their values. Like construction, replacement
assigns values without validating them against the class's annotations.

Batched structs
---------------

### Constructing a batch

A batch has the same fields as one element, with leading dimensions on the
arrays. For example, we can directly construct three particles:

```python
particles = Particle(
    position=jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]),
    velocity=jnp.array([[0.1, 0.0], [0.0, -0.1], [-0.1, 0.1]]),
    mass=jnp.array([1.0, 2.0, 1.0]),
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

You can define your struct for the individual elements of the batch, and then
annotate batched structs with the `strux.Struct` type form, following jaxtyping
dimensions syntax.

```python
Particles = strux.Struct[Particle, "particles"]

def total_mass(particles: Particles) -> Float[Array, ""]:
    return jnp.sum(particles.mass)

print(total_mass(particles))
```

Output:
```console
4.0
```

`Struct[Particle, "particles"]` describes positions and velocities with shape
`"particles 2"`, and masses with shape `"particles"`. The batch dimension is
shared by all three fields. Even a field that is scalar for one element
needs a batch dimension: strux does not broadcast it automatically.

The annotation describes the value; it doesn't create a new class of
particles or turn on runtime checking.

### Indexing and shape

Batched structs support `.shape`, `len`, indexing, and iteration. The batch
shape consists of the leading dimensions beyond each field's element shape.
For `particles`, those element shapes are `(2,)`, `(2,)`, and `()`.

```python
print(particles.shape)
print(len(particles))
print(particles[0])
print(particles[1:].shape)
print([float(p.mass) for p in particles])
```

Output:
```console
(3,)
3
Particle(
  position=jnp.float32[2],
  velocity=jnp.float32[2],
  mass=jnp.float32[],
)
(2,)
[1.0, 2.0, 1.0]
```

Indexing selects batch axes, leaving each element's vector and matrix axes
alone. An unbatched struct has shape `()` and refuses batch indexing.

### Network ensembles

Structs work naturally with vectorisation and `jax.vmap`.

We can initialise several networks by vmapping the element initialiser:

```python
keys = jax.random.split(jax.random.key(10), 3)
ensemble = jax.vmap(
    lambda key: MLP.init(key, features=4, hidden=8, outputs=1)
)(keys)
print(ensemble.shape)
print(ensemble.linear1.weights.shape)
print(ensemble[0])
```

Output:
```console
(3,)
(3, 4, 8)
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

The outer batch dimension appears on every array in the nested structure.
Static fields such as the activation function are shared by the ensemble.

Use `vmap` again to apply each network to the same input:

```python
@jax.jit
def predict_ensemble(
    ensemble: strux.Struct[MLP, "members"],
    x: Float[Array, "features"],
) -> Float[Array, "members outputs"]:
    return jax.vmap(MLP.forward, in_axes=(0, None))(ensemble, x)

predictions = predict_ensemble(ensemble, jnp.ones(4))
print(predictions.shape)
```

Output:
```console
(3, 1)
```

Methods remain ordinary Python methods. `forward` was written for one
network, so here we explicitly vmap it over the ensemble.

### Parallel rollouts

For environments, `vmap` advances several states in parallel and `scan`
collects successive states over time. Both operate directly on structs.

```python
BatchState = strux.Struct[GridWorld, "env"]
Trajectory = strux.Struct[GridWorld, "time env"]

@jax.jit
def batched_step(
    envs: BatchState,
    actions: Int[Array, "env"],
) -> BatchState:
    return jax.vmap(GridWorld.step)(envs, actions)

@jax.jit
def rollout(
    initial: BatchState,
    actions: Int[Array, "time env"],
) -> tuple[BatchState, Trajectory]:
    def body(envs, actions):
        next_envs = batched_step(envs, actions)
        return next_envs, next_envs

    return jax.lax.scan(body, initial, actions)

keys = jax.random.split(jax.random.key(0), 4)
envs = jax.vmap(GridWorld.init)(keys)
actions = jnp.tile(jnp.array([1, 2, 3, 4]), (6, 1))
final_envs, trajectory = rollout(envs, actions)

print(final_envs.shape)
print(trajectory.shape)
print(trajectory[0].shape)
print(trajectory[:, 0].shape)
print(trajectory[0, 0])
```

Output:
```console
(4,)
(6, 4)
(4,)
(6,)
GridWorld(
  hero_pos=jnp.int32[2],
  walls=jnp.bool[5,5],
)
```

`trajectory[0]` contains all environments after the first step;
`trajectory[:, 0]` contains the first environment's states across time.
The initial state is not included in this trajectory.

Annotation algebra
------------------

The same structure can hold parameters, masks, statistics, or batches of
any of these. `strux.Struct` describes those contents by transforming the
field annotations. These operations construct annotations; the numerical
operations on values are written separately using ordinary Python and JAX.

### Replacing field annotations

`strux.Replace` works like `.replace`, but for annotations. Named fields
receive new annotations; omitted fields keep their existing annotations.
For example, suppose we replace a layer's weight matrix with its mean over
inputs, keeping the biases unchanged:

```python
MeanWeights = strux.Struct[
    AffineTransform,
    strux.Replace(weights=Float[Array, "num_outputs"]),
]

def mean_weights(layer: AffineTransform) -> MeanWeights:
    return layer.replace(weights=jnp.mean(layer.weights, axis=0))

means = mean_weights(net.linear1)
print(means)
```

Output:
```console
AffineTransform(
  weights=jnp.float32[8],
  biases=jnp.float32[8],
)
```

`means` is still an `AffineTransform`. It satisfies a different description
of the fields. Methods still exist on it, but `forward` assumes a weight
matrix and is not an operation on these statistics.

We can replace multiple field annotations, including their dtypes:

```python
LayerMask = strux.Struct[
    AffineTransform,
    strux.Replace(
        weights=Bool[Array, "num_inputs num_outputs"],
        biases=Bool[Array, "num_outputs"],
    ),
]

def positive_entries(layer: AffineTransform) -> LayerMask:
    return jax.tree.map(lambda x: x > 0, layer)

mask = positive_entries(net.linear1)
print(mask)
```

Output:
```console
AffineTransform(
  weights=jnp.bool[4,8],
  biases=jnp.bool[8],
)
```

Descriptions also work in nested fields. This one changes the description
of just the second layer of the MLP:

```python
PartlySummarised = strux.Struct[
    MLP,
    strux.Replace(linear2=MeanWeights),
]

def summarise_second_layer(net: MLP) -> PartlySummarised:
    return net.replace(linear2=mean_weights(net.linear2))
```

### Mapping leaf annotations

When every numerical leaf undergoes the same kind of transformation, use
`strux.Map`. For example, here are two statistics per parameter array:

```python
two_statistics = strux.Map(lambda annotation: Float[Array, "2"])
MLPStats = strux.Struct[MLP, two_statistics]

def statistics(net: MLP) -> MLPStats:
    return jax.tree.map(
        lambda x: jnp.stack([jnp.mean(x), jnp.std(x)]),
        net,
    )

stats = statistics(net)
print(stats)
print(type(stats) is MLP)
```

Output:
```console
MLP(
  linear1=AffineTransform(
    weights=jnp.float32[2],
    biases=jnp.float32[2],
  ),
  linear2=AffineTransform(
    weights=jnp.float32[2],
    biases=jnp.float32[2],
  ),
  activate=<fn:relu>,
)
True
```

`Map` visits numerical leaf annotations through nested structs and
containers. Static fields are left alone. Its callback receives an
annotation, not an array; the callback above replaces each leaf annotation
with `Float[Array, "2"]` regardless of what was there before.

The resulting annotation can itself be a field annotation:

```python
@strux.struct
class Diagnostics:
    parameters: MLPStats
    loss: Float[Array, ""]

diagnostics = Diagnostics(parameters=stats, loss=jnp.array(0.5))
print(diagnostics.parameters.linear1.weights.shape)
```

Output:
```console
(2,)
```

### Batching as an annotation transformation

Prepending batch dimensions to each leaf is one example of a more general type
transformation. The familiar batch string is shorthand for mapping `Batch`
over the leaf annotations. These descriptions have the same meaning:

```python
Ensemble = strux.Struct[MLP, "members"]
AlsoEnsemble = strux.Struct[
    MLP,
    strux.Map(lambda annotation: strux.Batch[annotation, "members"]),
]
```

For example, `Batch[Float[Array, "inputs outputs"], "members"]` describes
an array with shape `"members inputs outputs"`. It retains the underlying
element requirements and adds the leading batch axes.

### Inserting a struct at a leaf

A leaf can also be replaced with a whole struct. Here we give the statistics
names instead of packing them into a two-element array:

```python
@strux.struct
class ArrayStats:
    mean: Float[Array, ""]
    std: Float[Array, ""]

NamedStats = strux.Struct[
    MLP,
    strux.Map(lambda annotation: ArrayStats),
]

def named_statistics(net: MLP) -> NamedStats:
    return jax.tree.map(
        lambda x: ArrayStats(mean=jnp.mean(x), std=jnp.std(x)),
        net,
    )

named_stats = named_statistics(net)
print(named_stats.linear1.weights)
```

Output:
```console
ArrayStats(
  mean=jnp.float32[],
  std=jnp.float32[],
)
```

The enclosing objects keep their classes and field names. The values at
those fields can have different structure: a field that held an array now
holds an `ArrayStats`.

### Composing annotations

`Struct` applies its transformations from left to right. An existing
annotation can be the starting point of another one:

```python
EnsembleStats = strux.Struct[MLPStats, "members"]
AlsoEnsembleStats = strux.Struct[MLP, two_statistics, "members"]

def ensemble_statistics(ensemble: Ensemble) -> EnsembleStats:
    return jax.vmap(statistics)(ensemble)

ensemble_stats = ensemble_statistics(ensemble)
print(ensemble_stats.linear1.weights.shape)
```

Output:
```console
(3, 2)
```

The two descriptions above are equivalent. In general:

```text
Struct[Struct[S, A], B] = Struct[S, A, B]
```

Order matters. Mapping every leaf to a two-element array and then batching
it produces `"members 2"`. Batching first and then replacing every leaf
annotation with `Float[Array, "2"]` describes just `"2"`.

Consecutive batch operations prepend dimensions, so these are equivalent:

```python
History = strux.Struct[MLPStats, "members", "time"]
AlsoHistory = strux.Struct[MLPStats, "time members"]
```

A subsequent map acts on the leaves of the description produced so far,
including leaves inside a newly inserted struct:

```python
StatisticFlags = strux.Struct[
    NamedStats,
    strux.Map(lambda annotation: Bool[Array, ""]),
]

flags = jax.tree.map(lambda x: x > 0, named_stats)
print(flags.linear1.weights)
```

Output:
```console
ArrayStats(
  mean=jnp.bool[],
  std=jnp.bool[],
)
```

The first map introduced `ArrayStats`; the second describes its `mean` and
`std` leaves as Booleans. A map does not repeatedly apply its own rule to
new leaves it has just introduced.

### Batch hints

Ordinary batch access uses the element shapes declared on the class. For
example, a weight matrix has two element axes, so additional leading axes
can be recognised as batch axes.

Our statistics have different element shapes. `ensemble_stats` contains
arrays of shape `(3, 2)`: three models, two statistics per array. The
original `MLP` and `AffineTransform` annotations don't describe that layout.
Tell strux how to index it with an explicit batch shape:

```python
stats_batch = ensemble_stats.as_batch(shape=(3,))
print(stats_batch.shape)
print(stats_batch[0].linear1.weights.shape)
print(stats_batch[1:].shape)
```

Output:
```console
(3,)
(2,)
(2,)
```

The hint changes how batch indexing works; it doesn't change the arrays or
check them against an annotation. It takes precedence over the default
shape inference. Checking against `EnsembleStats` is a separate operation.

JAX transformations can change shapes arbitrarily, so their outputs don't
inherit explicit hints. Apply a new hint to a transformed result when you
need one. Ordinary batch indexing updates the hint for the selected batch.

Type-checking
-------------

### Checking a value

Annotations can be used without any runtime checking. When you want to
check a value, call `.check(annotation)`. The check compares its fields,
including nested fields, with the requirements in that annotation.

```python
particle.check(strux.Struct[Particle, ""])
particles.check(Particles)
means.check(MeanWeights)
stats.check(MLPStats)
ensemble_stats.check(EnsembleStats)
```

Successful checks leave the values unchanged. A mismatch raises
`strux.ValidationError`:

```python
bad_particles = particles.replace(mass=jnp.ones(2))

try:
    bad_particles.check(Particles)
except strux.ValidationError:
    print("rejected!")
```

Output:
```console
rejected!
```

Construction and replacement are unchecked: creating `bad_particles` is
allowed. It is the subsequent check against `Particles` that rejects the
inconsistent batch dimensions.

With no argument, `.check()` uses the class's default field annotations,
with no additional batch dimensions. Thus `particle.check()` checks one
particle; use `particles.check(Particles)` to check a particle batch.

You can also construct transformed values directly and check their intended
description. No special construction path is needed:

```python
manual_stats = AffineTransform(
    weights=jnp.array([0.1, 0.2]),
    biases=jnp.array([0.0, 0.0]),
)
manual_stats.check(strux.Struct[AffineTransform, two_statistics])
```

Checking a value does not supply a batch hint. Use `.as_batch(...)` when
you need to establish how indexing should work.

### Checking function boundaries

Strux also works together with jaxtyping's runtime type checking. For example,
if you combine it with a typechecker like beartype, shape and dtype mismatches
are caught at function boundaries.

```python
from jaxtyping import jaxtyped
from beartype import beartype   # pip install beartype

@jaxtyped(typechecker=beartype)
def checked_step(
    envs: strux.Struct[GridWorld, "env"],
    actions: Int[Array, "env"],
) -> strux.Struct[GridWorld, "env"]:
    return jax.vmap(GridWorld.step)(envs, actions)

step_actions = jnp.array([1, 2, 3, 4])
checked_envs = checked_step(envs, step_actions)
print(checked_envs.shape)
```

Output:
```console
(4,)
```

The checker compares arguments and return values with their annotations.
Here `"env"` connects the number of states, the number of actions, and the
number of returned states. A call with two actions and four states fails:

```python
from jaxtyping import TypeCheckError

try:
    checked_step(envs, jnp.array([1, 2]))
except TypeCheckError:
    print("rejected!")
```

Output:
```console
rejected!
```

The same mechanism checks transformed descriptions. It can also check the
single-element function inside a vanilla `vmap`:

```python
@jaxtyped(typechecker=beartype)
def checked_statistics(net: strux.Struct[MLP, ""]) -> MLPStats:
    return statistics(net)

@jaxtyped(typechecker=beartype)
def checked_ensemble_statistics(ensemble: Ensemble) -> EnsembleStats:
    return jax.vmap(checked_statistics)(ensemble)

checked_stats = checked_ensemble_statistics(ensemble)
print(checked_stats.linear1.weights.shape)
```

Output:
```console
(3, 2)
```

These examples use explicit `Struct[...]` annotations to request recursive
field checks. Ordinary Python class membership alone doesn't check shapes
and dtypes. The decorators check function calls; constructors and `.replace`
remain available for assembling values with different contents.

Static type checkers currently see the underlying class in a `Struct[...]`
annotation. They do not check transformed leaf shapes and dtypes.

Serialisation
-------------

### Saving structs to disk

Structs can be saved to disk to be restored later, using the `save` method.

```python
import os, tempfile

# save the MLP to disk
path = os.path.join(tempfile.mkdtemp(), "mlp.npz")
net.save(path)
```

Two file formats are supported:

* If the filename has a `.npz` extension then the data fields are saved in
  compressed numpy format (pass `fmt="savez"` for uncompressed).

* If the filename has `.safetensors` extension, and strux is installed with the
  optional `safetensors` dependency (`pip install strux[safetensors]`), then
  strux uses the [safetensors](https://huggingface.co/docs/safetensors/)
  memory-mapped format.

By default, saving refuses to overwrite an existing file; pass `overwrite=True`
to replace it (e.g. for repeatedly saving the latest checkpoint during
training). Writes are atomic: data is written to a temporary file that is then
renamed over the destination, so an interrupted save never leaves a partial
file.

### Loading structs from disk

Structs saved to disk can be restored with the `restore` method of a
template instance, or with `strux.load`.

Restoration requires a template to determine the pytree structure.

1. One option is to supply an instance of the struct. The struct instance
   provides the pytree structure and the static field values, and the data
   leaves are read from the file. Strux raises an error if saved leaves don't
   match the template's shapes and dtypes.

   ```python
   # restore from disk using a fresh MLP as a template
   template = MLP.init(jax.random.key(999), features=4, hidden=8, outputs=1)
   restored = template.restore(path)

   # the restored model matches the original (not the template)
   print(jax.tree.all(jax.tree.map(jnp.array_equal, net, restored)))
   ```

   Output:
   ```console
   True
   ```

2. Alternatively, one can provide the struct class as a template. The shapes
   and dtypes are then taken from the file, along with some other static
   information (literal values for static fields involving python builtins or
   collections, union tags, etc.). Missing static information can be provided
   via a `statics=`, a dict keyed by `/`-separated field paths, or from
   defaults defined in the struct.

   ```python
   # instance-free restore: data fields rebuilt from the file, static fields
   # from statics= (or their defaults)
   restored = strux.load(
       path,
       template=MLP,
       statics={"activate": jax.nn.relu},
   )

   # matches
   print(jax.tree.all(jax.tree.map(jnp.array_equal, net, restored)))
   ```

   Output:
   ```console
   True
   ```

By default the format of the file is inferred from the file extension (use
`fmt=` to override).

### Inspecting checkpoints

You can inspect a saved struct without any template obligations using
`strux.describe`. This function inspects a saved file and renders its recorded
leaf data and structural information without constructing the actual structs.

```python
strux.describe(path)
```

Output (the header shows the path as given):
<!--output:illustrative-->
```console
mlp.npz (savez_compressed, strux format 2): 4 arrays, 49 elements, 196 B
__main__.MLP
  linear1: __main__.AffineTransform
    weights: float32[4 8]
    biases: float32[8]
  linear2: __main__.AffineTransform
    weights: float32[8 1]
    biases: float32[1]
```

The same functionality is available via the command line:

```console
$ python -m strux mlp.npz
mlp.npz (savez_compressed, strux format 2): 4 arrays, 49 elements, 196 B
__main__.MLP
  linear1: __main__.AffineTransform
    weights: float32[4 8]
    biases: float32[8]
  linear2: __main__.AffineTransform
    weights: float32[8 1]
    biases: float32[1]
```

### Other serialisation tools

You can also use raw `strux.to_dict` and `strux.from_dict` to convert structs
to and from flat dictionaries of arrays, with `strux.metadata` providing the
structure metadata that `save` would record (pass it back to `from_dict` as
`meta=`). This can compose with other Python serialisation tools.

### Orbax

Strux natively supports checkpointing with
[orbax](https://orbax.readthedocs.io/),
since structs are just pytrees:

```python
import orbax.checkpoint as ocp # pip install orbax-checkpoint

orbax_path = os.path.join(tempfile.mkdtemp(), "mlp_orbax")
with ocp.StandardCheckpointer() as checkpointer:
    # save (asynchronous: wait before handing the checkpoint to a reader)
    checkpointer.save(orbax_path, net)
    checkpointer.wait_until_finished()

    # restore
    restored = checkpointer.restore(orbax_path, target=template)

print(jax.tree.all(jax.tree.map(jnp.array_equal, net, restored)))
```

Output:
```console
True
```

<!-- Draft review notes (not part of the eventual published README):
- The proposed Map examples only need constant annotation replacement and
  Batch. The richer callback interface for inspecting/changing shape and
  dtype remains open; these examples do not select one.
- Map traversal through newly inserted structs is a proposed concrete rule:
  replacement is once per original leaf within a Map, and subsequent Maps
  traverse the result. It needs agreement before implementation.
- The optional .check() argument, default exact interpretation, exception
  name and no-return-value usage are draft API choices for review.
- as_batch storage and nested propagation remain open. These examples require
  explicit prefix indexing and slice updates but no automatic hint inference
  from successful checks.
- Full support for bare-class/Self boundary checking and global import hooks
  is not established by these explicit-Struct examples.
- Particle/rollout visualisations can be added after the examples are reviewed.
-->
