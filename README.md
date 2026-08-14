strux
=====

A JAX utility library for easily creating jit-able dataclasses.

Installation
------------

Install:

```console
pip install git+https://github.com/matomatical/strux.git
```

Requires python 3.12+. Dependencies: `jax`, `numpy`, `jaxtyping`.

Examples
--------

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

### Submodules and static fields

Structs can be nested arbitrarily, allowing one to easily implement complex
neural networks (among other things). For example, here is a multi-layer
perceptron module that combines two of the previous AffineTransform modules.

You can use the `static_fieldnames` flag for fields that shouldn't be traced
by JAX (e.g. configuration, shapes). These fields are excluded from `jax.jit`
and `jax.tree.map`. In the below example we use this to make the activation
function of the MLP configurable.


```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable, Self
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

Unlike in equinox, the decision of whether a field is static happens at class
definition time. This removes the need for filters and filtered JAX
transformations at a slight flexibility cost.

### Field annotations

What are all of these field annotations? A struct's field annotations describe
the elements of the struct. They are used by strux internally to support
validation, batch operations, and serialisation (described in later sections).

Data fields in particular come with some restrictions on their type
annotations. They should hold *array-leaved pytrees,* since these are the
things that can be traced. The most important supported annotations are as
follows.

* **Jaxtyping annotations** (e.g., `Float[Array, "n 2"]`): dtype kind and
  element dims as written. Dims may be concrete (`"5 5"`, checked exactly),
  symbolic (`"h w"` — each name must take one consistent size across the
  fields of the class, checked at construction; use the anonymous `"_"` for
  a dim that shouldn't bind), or unknown (`Float[Array, "..."]`, any element
  rank). Names are scoped per class: a nested struct's names bind at its own
  construction, so e.g. two layers of the same class may have different
  widths. See jaxtyping documentation for more information.
* **Plain scalars** (`float`, `int`, `bool`, `complex`): a Python scalar of
  that type, or a scalar array of the matching dtype kind. Equivalent to the
  explicit `Float[ArrayLike, ""]` spelling (`from jax.typing import
  ArrayLike`). A Python scalar is one element's worth of data: a *batched*
  struct holds an array in such a field (strux never broadcasts a scalar
  across a batch).
* **Unions**, including optional values `T | None`: the value decides the arm
  at construction. Note that which arm is instantiated is a *static* property:
  JIT recompiles per arm, traced control flow cannot switch arms.
* **Containers**: Containers such as `dict[str, T]`, `list[T]`, `tuple[T,
  ...]`, `tuple[T1, T2]`, as long as the leaves are supported.
* **Nested structs, other pytrees:** Nested structs are allowed. Actually, any
  other registered pytree is allowed, as long as the leaves are supported.

If you don't want to fully annotate your array types with jaxtyping, you can
use `jax.Array`. However, we don't allow `Any`.

```python
from jaxtyping import Array, Bool

@strux.struct
class Level:
    walls: Bool[Array, "size size"]  # jaxtyping annotation
    reward: float                    # plain scalar
    aux: jax.Array                   # bare array class

level = Level(
    walls=jnp.zeros((5, 5), dtype=bool),
    reward=1.0,
    aux=jnp.zeros(7),
)
print(level)
```

Output:
```console
Level(
  walls=jnp.bool[5,5],
  reward=float(1.0),
  aux=jnp.float32[7],
)
```

### Vmapping and batch annotations

Structs work naturally with vectorisation and `jax.vmap`.

You can define your struct for the individual elements of the batch, and then
annotate batched structs with the `strux.Struct` type form, following jaxtyping
dimensions syntax. For example, if `Image` is a struct, then:

* `Struct[Image, "batch_size"]` denotes a batched image.
* `Struct[Image, "b1 b2"]` denotes a batched batched image.
* `Struct[Image, "..."]` accepts any batch rank.
* `Struct[Image, ""]` is a rank-exact version (no batch dims).

We can use batch annotations to describe data batches, neural network
ensembles, or even depth-wise batches of layer parameters for use as inputs to
`jax.lax.scan`.  Here we give an example of a batching a gridworld for
collecting parallel rollouts.

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Bool, PRNGKeyArray
from typing import Self
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

# strux.Struct[GridWorld, "batch"] expands each field's annotation:
#   hero_pos: Int[Array, "batch 2"]
#   walls:    Bool[Array, "batch size size"]
def batched_step(
    envs: strux.Struct[GridWorld, "batch"],
    actions: Int[Array, "batch"],
) -> strux.Struct[GridWorld, "batch"]:
    return jax.vmap(GridWorld.step)(envs, actions)
```

Output:
```console
GridWorld(
  hero_pos=jnp.int32[4,2],
  walls=jnp.bool[4,5,5],
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

Note that plain scalar fields (annotated `float`, `int`, `bool`, `complex`) are
promoted to rank-0 jaxtyping annotations. For example, a `loss: float` field
batches as `Float[Array, "batch"]`.

### Indexing and shape for batched structs

Batched structs support `.shape`, `len`, indexing, and iteration. The `.shape`
property returns the batch dimensions (the leading dimensions beyond each
field's base annotation). Indexing with `env[i]` or slicing with `env[i:j]`
indexes into the batch dimensions of every field at once, `len(envs)` is the
leading batch dimension, and `for env in envs:` iterates over it.

Batch access is bounds-aware: integer indices are checked against the batch
shape (raising IndexError out of bounds, with negative indices in the python
style), and an *unbatched* struct refuses indexing, `len`, and iteration
outright (TypeError) — batch access never silently reaches into element
dimensions. Slices follow python slice semantics, and traced or array indices
are passed through to the leaves. The batch shape is solved once and cached on
the instance (structs are immutable), so these operations are cheap.

```python
# continuing from the previous example...

# .shape returns the batch dimensions
print(envs.shape)

# integer indexing extracts a single element from the batch
env0 = envs[0]
print(env0)

# slicing selects a range of elements from the batch
some_envs = envs[1:3]
print(some_envs.shape)
print(some_envs)
```

Output:
```console
(4,)
GridWorld(
  hero_pos=jnp.int32[2],
  walls=jnp.bool[5,5],
)
(2,)
GridWorld(
  hero_pos=jnp.int32[2,2],
  walls=jnp.bool[2,5,5],
)
```

These are convenience shortcuts for common operations on batched structs.
For other element-wise operations, use `jax.tree.map`:

```python
# adding a constant to every field isn't built in, but jax.tree.map works:
shifted = jax.tree.map(lambda x: x + 1, env0)
```

### Functor/map annotations

Prepending batch dimensions to each leaf is one example of a more general type
transformation: transforming the shape/datatype of leaves while preserving the
pytree structure. `strux.Struct` also allows us to annotate more general
transformed types. To explore, let's compare two checkpoints of the MLP from
the earlier examples: `net`, and a copy with one layer re-initialised.

```python
net2 = net.replace(
    linear2=AffineTransform.init(jax.random.key(1), 8, 1),
)
```

* Type casting: The annotation `strux.astype(dtype)` denotes every leaf keeping
  its dims but taking on the given dtype (`bool`, `int`, `float`, or
  `complex`). Useful for elementwise maps, like so:
  ```python
  # which entries changed? a bool mask per entry, shapes preserved
  def changed_entries(a: MLP, b: MLP) -> strux.Struct[MLP, strux.astype(bool)]:
      return jax.tree.map(lambda ai, bi: ai != bi, a, b)

  print(changed_entries(net, net2))
  ```

  Output:
  ```console
  MLP(
    linear1=AffineTransform(
      weights=jnp.bool[4,8],
      biases=jnp.bool[8],
    ),
    linear2=AffineTransform(
      weights=jnp.bool[8,1],
      biases=jnp.bool[1],
    ),
    activate=<fn:relu>,
  )
  ```

* Full reduction: The annotation `strux.mapped(dtype)` denotes every leaf
  becoming replaced with a scalar of the given dtype:
  ```python
  # how many entries changed in each field? one int per leaf
  def changed_counts(a: MLP, b: MLP) -> strux.Struct[MLP, strux.mapped(int)]:
      return jax.tree.map(lambda ai, bi: jnp.sum(ai != bi), a, b)

  counts = changed_counts(net, net2)
  print(counts.linear1.weights, counts.linear2.weights, counts.linear2.biases)
  ```

  Output:
  ```console
  0 8 0
  ```

  (Only the re-initialised layer's weights changed; its biases are zero in
  both checkpoints.)

* Axis reduction (planned): an annotation denoting every leaf reduced along a
  named element axis is designed but not yet implemented — see the roadmap.

`strux.Struct` takes a list of strings or the above functors and applies them
to the type of each leaf, left to right.


### Type validation on construction

By default, strux checks that field values match their type annotations at
construction time, and raises an error if there is a mismatch. (Pass
`@strux.struct(check=False)` to opt a class out of this feature.)

```python
# recall `Level` defined in the field annotations section above:

# batched construction: leading batch dims are free, but must agree
levels = Level(
    walls=jnp.zeros((32, 5, 5), dtype=bool),
    reward=jnp.ones(32),
    aux=jnp.zeros((32, 7)),
)
print(levels.shape)

# inconsistent batch dims are rejected at construction
try:
    Level(
        walls=jnp.zeros((32, 5, 5), dtype=bool),
        reward=jnp.ones(16),
        aux=jnp.zeros((32, 7)),
    )
except strux.ValidationError:
    print("rejected!")
```

Output:
```console
(32,)
rejected!
```

`.replace` is validated like direct construction, with one shortcut: a
replacement whose leaf layout (pytree structure, shapes, dtypes, python
scalar types) exactly matches the field it replaces cannot change the
instance's validity, so it skips revalidation — the common case of swapping
in same-shaped values (e.g. updated parameters each training step) costs no
solving.

The check costs microseconds per construction for simple schemas, and runs only
on direct construction and `.replace` (JAX's internal tree reconstructions skip
it).

Symbolic dim names (`"h w"`) are rank-only at construction;
`strux.tree_dims(obj)` binds them to sizes on demand and checks that shared
names agree across fields.

### Runtime type checking

Strux also works together with jaxtyping's runtime type checking. For example,
if you combine it with a typechecker like beartype, shape and dtype mismatches
are caught at function boundaries.

```python
from jaxtyping import jaxtyped
from beartype import beartype   # pip install beartype

@jaxtyped(typechecker=beartype)
def checked_step(
    envs: strux.Struct[GridWorld, "batch"], # GridWorld from previous example
    actions: Int[Array, "batch"],
) -> strux.Struct[GridWorld, "batch"]:
    return jax.vmap(GridWorld.step)(envs, actions)

# this passes: shapes and dtypes are consistent
envs = checked_step(envs, actions) # envs, actions from previous example

# this would fail: actions has wrong batch size
# checked_step(envs, jnp.array([1, 2]))  # beartype raises!
```

Function-boundary checking composes cleanly with strux's own construction
checking: constructors tolerate any (consistent) leading batch dims, and
discipline about a *specific* batch shape belongs at function boundaries,
as above.

External checkers that wrap dataclass constructors (such as jaxtyping's import
hook) find nothing to enforce on a struct's `__init__`: its runtime annotations
are deliberately empty, precisely so that hook-checked modules don't reject
legitimately batched constructions. Every other function in a hooked module is
checked as usual.

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

### More about serialisation

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

You can also use raw `strux.to_dict` and `strux.from_dict` to convert structs
to and from flat dictionaries of arrays, with `strux.metadata` providing the
structure metadata that `save` would record (pass it back to `from_dict` as
`meta=`). This can compose with other Python serialisation tools.

Strux natively supports checkpointing with
  [orbax](https://orbax.readthedocs.io/),
since structs are just pytrees:

```python
import orbax.checkpoint as ocp # pip install orbax-checkpoint

orbax_path = os.path.join(tempfile.mkdtemp(), "mlp_orbax")
checkpointer = ocp.StandardCheckpointer()

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

### Schema inference

Type checking (TODO: And structure serialisation) is supported internally by a
schema inferred from annotations. Inspect the inferred schema with
`strux.schema`:

```python
print(strux.schema(Level))
```

Output:
```console
schema Level:
  walls: Bool[Array, 'size size']
  reward: Float[Array, ''] | Float[ndarray, ''] | number | float
  aux: Shaped[Array, '...']
```

TODO: Describe schema inference.

Development
-----------

Development has some additional optional dependencies:

```
uv pip install -e ".[dev]"
```

Installs normal dependencies plus also `jaxtyping`, `beartype`, `pytest`,
`pytest-codeblocks`.

### Notes

The implementation is a small package (`strux/`), one module per component:
the `@struct` decorator (`struct.py`), schema compilation (`schema.py`), the
batch-shape solver (`batch.py`), shape queries and indexing (`shapes.py`),
the `Struct` type form and its functors (`annotate.py`), pretty printing
(`pprint.py`), and serialisation (`serial.py`). Tests mirror this layout in
`tests/`.

Jaxtyping is a required dependency: it is strux's annotation *language* —
the schema (`strux.schema`) is compiled from jaxtyping annotations, and
plain-scalar sugar expands through jaxtyping's ArrayLike support. It is a
featherweight dependency (it does not itself require JAX). Beartype is
*not* a dependency: construction checking is strux's own (schema-driven),
and beartype/`@jaxtyped` remain the recommended tools at function
boundaries only.

Reserved field names: fields named `replace`, `size`, `shape`, `save`, or
`restore` shadow the corresponding convenience member (strux warns and skips
adding it); the module-level equivalents (`dataclasses.replace`,
`strux.tree_size`, `strux.tree_shape`, `strux.save`, `strux.load`) always
remain available. In npz files the entry `__strux__` is reserved for strux
metadata, so a tree with a field of that name refuses to save as npz
(safetensors keeps metadata out of the array namespace and is unaffected).

### Versioning

From 0.1.0, versions and breaking changes are tracked in `CHANGELOG.md`.

### Testing

Run tests with `pytest`. Make sure this passes before committing, or at least
before merging to main.

The README's python blocks are executed in order and each `Output:` block is
compared against what the code actually prints (`tests/test_readme.py`).
Mark an output block with `<!--output:illustrative-->` to exempt it from the
comparison (used for outputs that cannot be deterministic, such as ones
containing temporary paths).

### Roadmap

Basics:

- [x] Frozen dataclass + JAX pytree registration via `@strux.struct` wrapper
- [x] Pretty printing with shape/dtype summaries for arrays
- [x] Static field support via `static_fieldnames`
- [x] Decorator syntax with keyword arguments (`@strux.struct(...)`)
- [x] Batched type annotations (`strux.Struct[MyStruct, "batch"]`; statically
      `typing.Annotated`, so legal and inert in checked signatures)

Advanced features:

- [x] `isinstance` support and integrate with jaxtyping + beartype
- [x] Save/load structs to/from disk (e.g. serialisation with pytree structure)
- [x] Support indexing and shape directly on batched structs, e.g., `env[0]`.
- [x] Schema-driven batch-tolerant construction checking (`strux.schema`)
- [x] Unions, optionals, containers, and unannotated arrays as data fields
- [x] Pretty print registered pytree classes that aren't dataclasses

Advanced serialisation:

- [x] Template-free restore: load from a struct class, no template instance
      (`strux.load(path, template=Cls, statics=...)`)
- [x] Serialise literal static fields where possible (`repr` /
      `ast.literal_eval` round-trip; no code execution on load; callables
      etc. are not attempted and still need `statics=` or a template)
- [x] Serialise instance-level structure aside from shapes/dtypes
      (subclass and union-arm tags in the safetensors metadata header / a
      reserved npz entry; more structs amenable to instance-free
      restoration; helps with checkpoint archaeology)
- [x] Error on loading a leaf with a different shape/dtype from the
      template (in analogy with spare/missing leaves, which error)
- [x] Checkpoint inspection without constructing structs
      (`strux.describe` / `python -m strux <checkpoint>`)

Advanced typing:

- [x] Generic structs (PEP 695): bounded-TypeVar and generic-alias field
      annotations (constraints erased to the bound/origin at runtime)
- [x] Schema functors: leaf-spec transformations as annotations —
      `strux.astype(kind)` and `strux.mapped(kind)`, composing with dims
      strings in the `Struct` form (the prepend being the invertible
      special case)
* Not happy with the whole strux relationship to annotations / types, since we
  can't check the result of maps for example.
- [ ] More functors as needs arise: `reduced(*names)` (drop named element
      axes) and `promoted()` (sum-semantics dtype promotion) are designed
      in the meta-repo journal
- [x] Bind symbolic dim names across fields (enforced at construction,
      scoped per class, `"_"` anonymous dims exempt; `strux.tree_dims`
      queries the bindings; `Struct`-form isinstance checks additionally
      require a consistent cross-field batch solution)

Project:

- [x] Unit tests
- [x] README example tests
- [ ] Type annotations and static type checking
- [ ] Documentation
- [ ] List on PyPI
