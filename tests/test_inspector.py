"""
Tests for checkpoint inspection: strux.describe and the python -m strux
entry point.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

import strux
from strux.__main__ import main


@strux.struct
class Affine:
    weights: Float[Array, "n_in n_out"]
    biases: Float[Array, "n_out"]


@strux.struct(static_fieldnames=("activate",))
class Net:
    linear: Affine
    momentum: Float[Array, "n_out"] | None
    activate: str


def _make_net():
    return Net(
        linear=Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(8)),
        momentum=None,
        activate="relu",
    )


class TestDescribe:
    def test_structure_and_metadata_shown(self, tmp_path):
        path = str(tmp_path / "net.npz")
        _make_net().save(path)
        description = strux.describe(path)
        assert "strux format 2" in description
        assert ".Net" in description         # root class tag
        assert ".Affine" in description      # nested class tag
        assert "weights: float32[4 8]" in description
        assert "activate: 'relu' (static)" in description
        assert "momentum: None" in description

    def test_recorded_dtypes_shown(self, tmp_path):
        @strux.struct
        class Weights:
            w: Float[Array, "n"]

        path = str(tmp_path / "w.npz")
        Weights(w=jnp.ones(3, dtype=jnp.bfloat16)).save(path)
        assert "w: bfloat16[3]" in strux.describe(path)

    def test_foreign_file_described(self, tmp_path):
        path = str(tmp_path / "foreign.npz")
        np.savez(path, alpha=np.ones((2, 3), dtype=np.float32))
        description = strux.describe(path)
        assert "no strux metadata" in description
        assert "alpha: float32[2 3]" in description

    def test_safetensors_described(self, tmp_path):
        path = str(tmp_path / "net.safetensors")
        _make_net().save(path)
        description = strux.describe(path)
        assert "strux format 2" in description
        assert "weights: float32[4 8]" in description


class TestMain:
    def test_describes_file(self, tmp_path, capsys):
        path = str(tmp_path / "net.npz")
        _make_net().save(path)
        assert main([path]) == 0
        assert "float32[4 8]" in capsys.readouterr().out

    def test_usage_on_bad_args(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out
        assert main(["--help"]) == 0
