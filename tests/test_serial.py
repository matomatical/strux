"""
Tests for serialisation: dict flattening (to_dict/from_dict), save/load in
npz and safetensors formats, and template-free restore.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int

import strux

from example_structs import Environment, Point, World


def _assert_equal(a, b):
    """Assert two arrays have equal values AND dtypes."""
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} != {b.dtype}"
    assert jnp.array_equal(a, b)


def _make_env():
    return Environment(
        hero_pos=jnp.array([1, 2], dtype=jnp.int32),
        goal_pos=jnp.array([3, 4], dtype=jnp.int32),
        walls=jnp.ones((5, 5), dtype=bool),
    )


def _make_world():
    return World(env=_make_env(), score=jnp.float32(42.0))


class TestToDict:
    def test_flat_struct_keys(self):
        d = strux.to_dict(_make_env())
        assert set(d.keys()) == {"hero_pos", "goal_pos", "walls"}

    def test_nested_struct_keys(self):
        d = strux.to_dict(_make_world())
        assert set(d.keys()) == {"env/hero_pos", "env/goal_pos", "env/walls", "score"}

    def test_values_are_numpy(self):
        import numpy
        d = strux.to_dict(_make_env())
        for v in d.values():
            assert isinstance(v, numpy.ndarray)

    def test_dict_keys_use_repr(self):
        tree = {"a/b": jnp.array(1.0), "c": jnp.array(2.0)}
        d = strux.to_dict(tree)
        assert set(d.keys()) == {"'a/b'", "'c'"}

    def test_sequence_keys_use_repr(self):
        tree = [jnp.array(1.0), jnp.array(2.0)]
        d = strux.to_dict(tree)
        assert set(d.keys()) == {"0", "1"}

    def test_mixed_tree_keys(self):
        tree = {"params": _make_env(), "steps": [jnp.array(1), jnp.array(2)]}
        d = strux.to_dict(tree)
        assert "'params'/hero_pos" in d
        assert "'steps'/0" in d

    def test_key_clash_raises(self):
        class Evil(str):
            def __repr__(self):
                return "'a'"
        tree = {"a": jnp.array(1.0), Evil("b"): jnp.array(2.0)}
        with pytest.raises(ValueError, match="Key clash"):
            strux.to_dict(tree)


class TestFromDict:
    def test_round_trip(self):
        original = _make_world()
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_round_trip_dict_tree(self):
        original = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored["a"], original["a"])
        _assert_equal(restored["b"], original["b"])

    def test_round_trip_list_tree(self):
        original = [jnp.array(1.0), jnp.array(2.0)]
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored[0], original[0])
        _assert_equal(restored[1], original[1])

    def test_round_trip_mixed_tree(self):
        original = {"params": _make_env(), "step": jnp.array(0)}
        d = strux.to_dict(original)
        restored = strux.from_dict(d, template=original)
        _assert_equal(restored["params"].hero_pos, original["params"].hero_pos)
        _assert_equal(restored["step"], original["step"])

    def test_missing_key_raises(self):
        d = {"hero_pos": jnp.zeros(2), "goal_pos": jnp.zeros(2)}
        with pytest.raises(KeyError, match="walls"):
            strux.from_dict(d, template=_make_env())

    def test_extra_keys_raises(self):
        d = strux.to_dict(_make_env())
        d["extra"] = jnp.zeros(3)
        with pytest.raises(KeyError, match="extra keys"):
            strux.from_dict(d, template=_make_env())

    def test_static_fields_from_template(self):
        @strux.struct(static_fieldnames=("label",))
        class Labelled:
            pos: Int[Array, "2"]
            label: str
        template = Labelled(pos=jnp.zeros(2, dtype=jnp.int32), label="hello")
        d = {"pos": jnp.array([10, 20], dtype=jnp.int32)}
        restored = strux.from_dict(d, template=template)
        _assert_equal(restored.pos, jnp.array([10, 20]))
        assert restored.label == "hello"


class TestSaveLoadNpz:
    def test_flat_struct(self, tmp_path):
        original = _make_env()
        path = tmp_path / "env.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Environment)
        _assert_equal(restored.hero_pos, original.hero_pos)
        _assert_equal(restored.walls, original.walls)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.walls, original.env.walls)


class TestSaveLoadSafetensors:
    def test_flat_struct(self, tmp_path):
        original = Point(x=jnp.float32(1.0), y=jnp.float32(2.0))
        path = tmp_path / "point.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, Point)
        _assert_equal(restored.x, original.x)
        _assert_equal(restored.y, original.y)

    def test_nested_struct(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.safetensors"
        strux.save(path, original)
        restored = strux.load(path, template=original)
        assert isinstance(restored, World)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)


class TestSaveLoadErrors:
    def test_unknown_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot infer format"):
            strux.save(tmp_path / "file.xyz", _make_env())

    def test_explicit_format_overrides_extension(self, tmp_path):
        path = tmp_path / "file.npz"
        strux.save(path, _make_env(), fmt="savez")
        restored = strux.load(path, template=_make_env(), fmt="savez")
        _assert_equal(restored.hero_pos, _make_env().hero_pos)

    def test_overwrite_raises(self, tmp_path):
        path = tmp_path / "env.npz"
        strux.save(path, _make_env())
        with pytest.raises(FileExistsError, match="already exists"):
            strux.save(path, _make_env())

    def test_overwrite_true_replaces(self, tmp_path):
        path = tmp_path / "env.npz"
        original = _make_env()
        strux.save(path, original)
        updated = original.replace(hero_pos=original.hero_pos + 1)
        strux.save(path, updated, overwrite=True)
        restored = strux.load(path, template=original)
        _assert_equal(restored.hero_pos, updated.hero_pos)

    def test_no_temporary_files_left_behind(self, tmp_path):
        # saves go via a temporary file renamed over the destination
        path = tmp_path / "env.npz"
        strux.save(path, _make_env())
        strux.save(path, _make_env(), overwrite=True)
        assert [p.name for p in tmp_path.iterdir()] == ["env.npz"]

    def test_npz_defaults_to_compressed(self, tmp_path):
        # use a large zero array so compression is clearly effective
        @strux.struct
        class Big:
            data: Float[Array, "n"]
        big = Big(data=jnp.zeros(10_000))
        path_default = tmp_path / "default.npz"
        path_explicit = tmp_path / "explicit.npz"
        path_uncompressed = tmp_path / "uncompressed.npz"
        strux.save(path_default, big)                              # default
        strux.save(path_explicit, big, fmt="savez_compressed")  # explicit
        strux.save(path_uncompressed, big, fmt="savez")         # uncompressed
        # all three round-trip correctly
        for p in (path_default, path_explicit, path_uncompressed):
            restored = strux.load(p, template=big)
            _assert_equal(restored.data, big.data)
        # default and explicit compressed produce the same file size
        assert path_default.stat().st_size == path_explicit.stat().st_size
        # compressed is strictly smaller than uncompressed
        assert path_default.stat().st_size < path_uncompressed.stat().st_size


class TestSaveRestoreMethods:
    def test_save_and_restore(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        original.save(path)
        restored = original.restore(path)
        assert isinstance(restored, World)
        _assert_equal(restored.score, original.score)
        _assert_equal(restored.env.hero_pos, original.env.hero_pos)

    def test_save_method_overwrite(self, tmp_path):
        original = _make_world()
        path = tmp_path / "world.npz"
        original.save(path)
        original.save(path, overwrite=True)
        restored = original.restore(path)
        _assert_equal(restored.score, original.score)

    def test_save_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'save'"):
            @strux.struct
            class HasSave:
                save: int
                x: float

    def test_restore_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'restore'"):
            @strux.struct
            class HasRestore:
                restore: int
                x: float


# # #
# Template-free restore (load from a struct class)


class TestTemplateFreeRestore:
    def _mlp(self):
        # a nested module with a static field, like the README MLP
        @strux.struct
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        @strux.struct(static_fieldnames=("activate",))
        class MLP:
            linear1: Affine
            linear2: Affine
            activate: object

        net = MLP(
            linear1=Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(8)),
            linear2=Affine(weights=jnp.ones((8, 1)), biases=jnp.zeros(1)),
            activate=jax.nn.relu,
        )
        return MLP, net

    def test_nested_restore_with_statics(self, tmp_path):
        MLP, net = self._mlp()
        path = str(tmp_path / "mlp.npz")
        net.save(path)
        restored = strux.load(
            path, template=MLP, statics={"activate": jax.nn.relu},
        )
        assert jax.tree.all(jax.tree.map(jnp.array_equal, net, restored))
        assert restored.activate is jax.nn.relu

    def test_missing_static_names_the_path(self, tmp_path):
        MLP, net = self._mlp()
        path = str(tmp_path / "mlp.npz")
        net.save(path)
        with pytest.raises(KeyError, match="statics=\\{'activate'"):
            strux.load(path, template=MLP)

    def test_static_recorded_literal_used(self, tmp_path):
        # literal statics are recorded in the file and win over defaults
        @strux.struct(static_fieldnames=("name",))
        class Tagged:
            x: Float[Array, ""]
            name: str = "anon"

        t = Tagged(x=jnp.float32(1.0), name="custom")
        path = str(tmp_path / "tagged.npz")
        t.save(path)
        restored = strux.load(path, template=Tagged)
        assert restored.name == "custom"    # the saved instance's, not the default
        assert jnp.array_equal(restored.x, t.x)

    def test_static_default_used_without_metadata(self):
        # with no metadata (a file from an earlier strux, or a bare dict),
        # statics fall back to their defaults
        @strux.struct(static_fieldnames=("name",))
        class Tagged:
            x: Float[Array, ""]
            name: str = "anon"

        t = Tagged(x=jnp.float32(1.0), name="custom")
        restored = strux.from_dict(strux.to_dict(t), template=Tagged)
        assert restored.name == "anon"

    def test_statics_param_overrides_recorded_literal(self, tmp_path):
        @strux.struct(static_fieldnames=("name",))
        class Tagged:
            x: Float[Array, ""]
            name: str = "anon"

        t = Tagged(x=jnp.float32(1.0), name="custom")
        path = str(tmp_path / "tagged.npz")
        t.save(path)
        restored = strux.load(path, template=Tagged, statics={"name": "mine"})
        assert restored.name == "mine"

    def test_optional_field_roundtrip(self, tmp_path):
        @strux.struct
        class Opt:
            momentum: Float[Array, "n"] | None
            count: Int[Array, ""]

        for momentum in (None, jnp.arange(3.0)):
            o = Opt(momentum=momentum, count=jnp.int32(7))
            path = str(tmp_path / f"opt_{momentum is None}.npz")
            o.save(path)
            restored = strux.load(path, template=Opt)
            assert (restored.momentum is None) == (momentum is None)
            assert restored.count == 7

    def test_container_fields_roundtrip(self, tmp_path):
        @strux.struct
        class Bank:
            layers: tuple[Float[Array, "_"], ...]   # "_": ragged elements allowed
            table: dict[str, Float[Array, "2"]]

        bank = Bank(
            layers=(jnp.zeros(3), jnp.ones(5)),
            table={"a": jnp.zeros(2), "b": jnp.ones(2)},
        )
        path = str(tmp_path / "bank.npz")
        bank.save(path)
        restored = strux.load(path, template=Bank)
        assert jax.tree.all(jax.tree.map(jnp.array_equal, bank, restored))
        assert isinstance(restored.layers, tuple)
        assert set(restored.table) == {"a", "b"}

    def test_polymorphic_field_restored_from_class_tag(self, tmp_path):
        # a base-annotated field holding a subclass instance: the saved
        # class tag resolves the subclass (among imported classes)
        class RewardFn:
            pass

        @strux.struct
        class Constant(RewardFn):
            value: Float[Array, ""]

        @strux.struct
        class Holder:
            fn: RewardFn

        h = Holder(fn=Constant(value=jnp.float32(1.0)))
        path = str(tmp_path / "holder.npz")
        h.save(path)
        restored = strux.load(path, template=Holder)
        assert type(restored.fn) is Constant
        assert jnp.array_equal(restored.fn.value, h.fn.value)
        # and the instance template also works
        restored = strux.load(path, template=h)
        assert jnp.array_equal(restored.fn.value, h.fn.value)

    def test_polymorphic_field_refused_without_metadata(self):
        # with no metadata the subclass is unknowable: refuse
        class RewardFn:
            pass

        @strux.struct
        class Constant(RewardFn):
            value: Float[Array, ""]

        @strux.struct
        class Holder:
            fn: RewardFn

        h = Holder(fn=Constant(value=jnp.float32(1.0)))
        with pytest.raises((KeyError, TypeError), match="instance template"):
            strux.from_dict(strux.to_dict(h), template=Holder)

    def test_restored_batched_checkpoint(self, tmp_path):
        # a batched struct restores with its batch dims (and validation)
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        path = str(tmp_path / "points.npz")
        batched.save(path)
        restored = strux.load(path, template=Point)
        assert restored.shape == (4,)

    def test_statics_rejected_for_instance_template(self, tmp_path):
        p = Point(x=jnp.float32(0.0), y=jnp.float32(1.0))
        path = str(tmp_path / "p.npz")
        p.save(path)
        with pytest.raises(TypeError, match="instance template"):
            strux.load(path, template=p, statics={"x": 1})


# # #
# Union arm inference: sound only when unambiguous


@strux.struct
class _Metric:
    value: Float[Array, ""]


@strux.struct
class _Score:  # key-layout-isomorphic to _Metric
    value: Float[Array, ""]


@strux.struct
class _Pos:  # distinguishable from both by key layout
    coords: Float[Array, "2"]


class TestUnionArmInference:
    def test_isomorphic_arms_resolved_by_recorded_tag(self, tmp_path):
        # two arms with identical key layouts: the saved class tag records
        # which arm was in play, so template-free restore picks correctly
        @strux.struct
        class Holder:
            item: _Metric | _Score

        h = Holder(item=_Score(value=jnp.float32(3.0)))
        path = str(tmp_path / "h.npz")
        h.save(path)
        restored = strux.load(path, template=Holder)
        assert type(restored.item) is _Score

    def test_isomorphic_arms_raise_without_metadata(self):
        # with no metadata the arms are indistinguishable in the saved
        # arrays: refuse rather than silently pick one
        @strux.struct
        class Holder:
            item: _Metric | _Score

        h = Holder(item=_Score(value=jnp.float32(3.0)))
        with pytest.raises(KeyError, match="more than one arm"):
            strux.from_dict(strux.to_dict(h), template=Holder)

    def test_isomorphic_arms_instance_template_ok(self, tmp_path):
        @strux.struct
        class Holder:
            item: _Metric | _Score

        h = Holder(item=_Score(value=jnp.float32(3.0)))
        path = str(tmp_path / "h.npz")
        h.save(path)
        restored = strux.load(path, template=h)
        assert type(restored.item) is _Score

    def test_distinguishable_arms_restore(self, tmp_path):
        @strux.struct
        class Holder:
            item: _Pos | _Metric

        for item in (_Metric(value=jnp.float32(1.0)),
                     _Pos(coords=jnp.zeros(2))):
            h = Holder(item=item)
            path = str(tmp_path / f"h_{type(item).__name__}.npz")
            h.save(path)
            restored = strux.load(path, template=Holder)
            assert type(restored.item) is type(item)

    def test_subset_arms_choose_full_explanation(self, tmp_path):
        # one arm's keys are a subset of another's: the unique arm that
        # explains every saved key wins, regardless of declaration order
        @strux.struct
        class Both:
            value: Float[Array, ""]
            extra: Float[Array, ""]

        @strux.struct
        class Holder:
            item: _Metric | Both   # _Metric's keys ⊂ Both's keys

        h = Holder(item=Both(value=jnp.float32(1.0),
                             extra=jnp.float32(2.0)))
        path = str(tmp_path / "h.npz")
        h.save(path)
        restored = strux.load(path, template=Holder)
        assert type(restored.item) is Both


# # #
# Recorded dtypes: ml_dtypes leaves survive npz


class TestRecordedDtypes:
    @pytest.mark.parametrize("dtype_name", ["bfloat16", "float8_e4m3fn"])
    def test_mldtype_npz_roundtrip(self, tmp_path, dtype_name):
        # npz stores ml_dtypes as raw bytes; the recorded dtype views them
        # back on load
        @strux.struct
        class Weights:
            w: Float[Array, "n"]

        original = Weights(w=jnp.ones(3, dtype=jnp.dtype(dtype_name)))
        path = str(tmp_path / "w.npz")
        original.save(path)
        for template in (original, Weights):
            restored = strux.load(path, template=template)
            assert restored.w.dtype == jnp.dtype(dtype_name)
            assert jnp.array_equal(restored.w, original.w)

    def test_bfloat16_safetensors_roundtrip(self, tmp_path):
        # safetensors records ml_dtypes natively: no view needed
        @strux.struct
        class Weights:
            w: Float[Array, "n"]

        original = Weights(w=jnp.ones(3, dtype=jnp.bfloat16))
        path = str(tmp_path / "w.safetensors")
        original.save(path)
        restored = strux.load(path, template=original)
        assert restored.w.dtype == jnp.bfloat16


# # #
# Strict restore: instance templates refuse mismatched leaves


@strux.struct
class _Linear:
    weights: Float[Array, "n_in n_out"]
    biases: Float[Array, "n_out"]


class TestStrictRestore:
    def test_shape_mismatch_raises(self, tmp_path):
        saved = _Linear(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        path = str(tmp_path / "lin.npz")
        saved.save(path)
        template = _Linear(weights=jnp.ones((4, 16)), biases=jnp.zeros(16))
        with pytest.raises(ValueError, match="strict restore"):
            strux.load(path, template=template)

    def test_dtype_mismatch_raises(self, tmp_path):
        saved = _Linear(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        path = str(tmp_path / "lin.npz")
        saved.save(path)
        template = _Linear(
            weights=jnp.ones((4, 8), dtype=jnp.bfloat16),
            biases=jnp.zeros(8, dtype=jnp.bfloat16),
        )
        with pytest.raises(ValueError, match="bfloat16"):
            strux.load(path, template=template)

    def test_all_mismatches_reported(self, tmp_path):
        saved = _Linear(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        path = str(tmp_path / "lin.npz")
        saved.save(path)
        template = _Linear(weights=jnp.ones((4, 16)), biases=jnp.zeros(16))
        with pytest.raises(ValueError) as err:
            strux.load(path, template=template)
        assert "'weights'" in str(err.value)
        assert "'biases'" in str(err.value)

    def test_matching_template_restores(self, tmp_path):
        saved = _Linear(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        path = str(tmp_path / "lin.npz")
        saved.save(path)
        template = _Linear(weights=jnp.zeros((4, 8)), biases=jnp.ones(8))
        restored = strux.load(path, template=template)
        assert jax.tree.all(jax.tree.map(jnp.array_equal, saved, restored))

    def test_python_scalar_leaf_keeps_its_type(self, tmp_path):
        @strux.struct
        class Config:
            x: Float[Array, ""]
            rate: float

        saved = Config(x=jnp.float32(1.0), rate=0.5)
        path = str(tmp_path / "c.npz")
        saved.save(path)
        restored = strux.load(path, template=saved)
        assert type(restored.rate) is float
        assert restored.rate == 0.5

    def test_isomorphic_arm_template_mismatch_raises(self, tmp_path):
        # the checkpoint records which arm was saved, so restoring into a
        # template holding the *other*, key-layout-identical arm is caught
        @strux.struct
        class Holder:
            item: _Metric | _Score

        saved = Holder(item=_Score(value=jnp.float32(3.0)))
        path = str(tmp_path / "h.npz")
        saved.save(path)
        template = Holder(item=_Metric(value=jnp.float32(0.0)))
        with pytest.raises(TypeError, match="recorded structure disagrees"):
            strux.load(path, template=template)


# # #
# Structure tags: recorded classes and arms


class TestStructureTags:
    def test_root_subclass_restored(self, tmp_path):
        @strux.struct
        class Base:
            x: Float[Array, ""]

        @strux.struct
        class Extended(Base):
            y: Float[Array, ""]

        saved = Extended(x=jnp.float32(1.0), y=jnp.float32(2.0))
        path = str(tmp_path / "e.npz")
        saved.save(path)
        restored = strux.load(path, template=Base)
        assert type(restored) is Extended
        assert jnp.array_equal(restored.y, saved.y)

    def test_unresolvable_class_tag_raises(self):
        @strux.struct
        class Alone:
            x: Float[Array, ""]

        d = strux.to_dict(Alone(x=jnp.float32(1.0)))
        meta = {"strux": "2", "class": "ghost_module.GhostClass"}
        with pytest.raises(TypeError, match="import the module"):
            strux.from_dict(d, template=Alone, meta=meta)

    def test_recorded_arm_missing_from_schema_raises(self):
        @strux.struct
        class Opt:
            momentum: Float[Array, "n"] | None

        o = Opt(momentum=None)
        d = strux.to_dict(o)
        meta = {"strux": "2", "arm momentum": "dict"}
        with pytest.raises(TypeError, match="no matching arm"):
            strux.from_dict(d, template=Opt, meta=meta)

    def test_none_arm_tag_roundtrip(self, tmp_path):
        # a None field leaves no arrays behind; the arm tag records it
        @strux.struct
        class Opt:
            momentum: Float[Array, "n"] | None
            count: Int[Array, ""]

        o = Opt(momentum=None, count=jnp.int32(7))
        path = str(tmp_path / "o.npz")
        o.save(path)
        restored = strux.load(path, template=Opt)
        assert restored.momentum is None

    def test_npz_reserved_key_collision_raises(self, tmp_path):
        # a field named __strux__ collides with the reserved npz metadata
        # entry (dict keys are repr-quoted in paths, so they never do)
        @strux.struct
        class Clash:
            __strux__: Float[Array, ""]

        c = Clash(__strux__=jnp.float32(1.0))
        path = str(tmp_path / "clash.npz")
        with pytest.raises(ValueError, match="reserved"):
            strux.save(path, c)
        # the safetensors header is separate from the arrays: no collision
        c.save(str(tmp_path / "clash.safetensors"))


# # #
# Public metadata: the dict-level companion of to_dict/from_dict


class TestPublicMetadata:
    def test_dict_level_roundtrip_with_metadata(self):
        @strux.struct(static_fieldnames=("name",))
        class Tagged:
            x: Float[Array, ""]
            name: str

        t = Tagged(x=jnp.float32(1.0), name="custom")
        d = strux.to_dict(t)
        meta = strux.metadata(t)
        assert meta["strux"] == "2"
        restored = strux.from_dict(d, template=Tagged, meta=meta)
        assert restored.name == "custom"
        assert jnp.array_equal(restored.x, t.x)
