strux
=====

A rough JAX utility library for easily creating jit-able dataclasses.

Install:

```console
pip install git+https://github.com/matomatical/strux.git
```

Quick start:

```python
import functools
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import strux

@strux.struct
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("num_inputs", "num_outputs"))
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

key = jax.random.key(seed=42)
net = AffineTransform.init(key=key, num_inputs=10, num_outputs=1)
print(net)
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
