"""Restoration preserves recorded configuration and fills only missing statics."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import strux


@strux.struct(static_fieldnames=("settings", "activate"))
class Module:
    weights: Float[Array, "n"]
    settings: object = "default"
    activate: object = jax.nn.relu


@pytest.mark.parametrize("suffix", ["npz", "safetensors"])
def test_instance_statics_must_match_checkpoint(suffix, tmp_path):
    saved = Module(jnp.ones(2), settings={"normalise": True, "widths": [2, 3]})
    path = tmp_path / f"module.{suffix}"
    saved.save(path)
    template = saved.replace(weights=jnp.zeros(2))
    restored = strux.load(path, template=template)
    assert restored.settings is template.settings
    assert jnp.array_equal(restored.weights, saved.weights)
    with pytest.raises(ValueError, match="at settings: checkpoint has"):
        strux.load(path, template=template.replace(settings={"normalise": False}))


@pytest.mark.parametrize(
    "saved_value, template_value",
    [
        (None, "default"),
        (True, 1),
        ([1], [True]),
        ({"enabled": True}, {"enabled": 1}),
        ({True: "value"}, {1: "value"}),
        ({1}, {True}),
        ((1, 2), [1, 2]),
        ("literal", jax.nn.relu),
        (1, jnp.array(1)),
    ],
)
def test_static_comparison_preserves_literal_types(saved_value, template_value):
    saved = Module(jnp.ones(2), settings=saved_value)
    with pytest.raises(ValueError, match="recorded static values disagree"):
        strux.from_dict(
            strux.to_dict(saved),
            template=saved.replace(settings=template_value),
            meta=strux.metadata(saved),
        )


def test_equal_collections_need_not_have_identical_repr():
    saved = Module(jnp.ones(2), settings={"a": [1, 2], "b": {3, 4}})
    settings = {"b": {4, 3}, "a": [1, 2]}
    arrays = strux.to_dict(saved)
    meta = strux.metadata(saved)
    restored = strux.from_dict(
        arrays, template=saved.replace(settings=settings), meta=meta
    )
    assert restored.settings is settings
    restored = strux.from_dict(
        arrays, template=Module, statics={"settings": settings}, meta=meta
    )
    assert restored.settings == saved.settings


def test_nested_static_conflicts_report_all_paths():
    @strux.struct
    class Ensemble:
        members: dict[str, Module]

    saved = Ensemble(
        {
            "left": Module(jnp.ones(2), settings="saved"),
            "right": Module(jnp.ones(2), settings="saved"),
        }
    )
    template = Ensemble(
        {key: value.replace(settings="other") for key, value in saved.members.items()}
    )
    with pytest.raises(ValueError) as caught:
        strux.from_dict(
            strux.to_dict(saved), template=template, meta=strux.metadata(saved)
        )
    assert "members/'left'/settings" in str(caught.value)
    assert "members/'right'/settings" in str(caught.value)
    with pytest.raises(ValueError, match="supplied statics disagree"):
        strux.from_dict(
            strux.to_dict(saved),
            template=Ensemble,
            statics={"members/'left'/settings": "other"},
            meta=strux.metadata(saved),
        )


def test_unrecorded_statics_come_from_template_or_supplied_values():
    saved = Module(jnp.ones(2), settings="saved", activate=jax.nn.relu)
    arrays = strux.to_dict(saved)
    meta = strux.metadata(saved)
    assert "static activate" not in meta
    restored = strux.from_dict(
        arrays, template=saved.replace(activate=jnp.sin), meta=meta
    )
    assert restored.activate is jnp.sin
    restored = strux.from_dict(
        arrays, template=Module, statics={"activate": jnp.sin}, meta=meta
    )
    assert restored.activate is jnp.sin
    assert restored.settings == "saved"


def test_class_restore_prefers_recorded_statics_then_supplied_then_defaults():
    saved = Module(jnp.ones(2), settings="saved")
    arrays = strux.to_dict(saved)
    meta = strux.metadata(saved)
    assert strux.from_dict(arrays, template=Module, meta=meta).settings == "saved"
    assert strux.from_dict(arrays, template=Module).settings == "default"
    assert (
        strux.from_dict(
            arrays, template=Module, statics={"settings": "supplied"}
        ).settings
        == "supplied"
    )
    assert (
        strux.from_dict(
            arrays, template=Module, statics={"settings": "saved"}, meta=meta
        ).settings
        == "saved"
    )
    with pytest.raises(ValueError, match="at settings"):
        strux.from_dict(
            arrays, template=Module, statics={"settings": "other"}, meta=meta
        )
    template = saved.replace(settings="template")
    assert strux.from_dict(arrays, template=template).settings == "template"


def test_static_check_does_not_require_a_compilable_data_schema():
    # Instance restoration is also available for arbitrary transformed contents.
    @strux.struct(static_fieldnames=("label",))
    class Opaque:
        data: object
        label: str

    template = Opaque(jnp.ones(2), label="template")
    with pytest.raises(ValueError, match="at label"):
        strux.from_dict(
            strux.to_dict(template), template=template, meta={"static label": "'saved'"}
        )


@pytest.mark.parametrize("use_class", [False, True])
def test_recorded_statics_missing_from_template_are_rejected(use_class):
    saved = Module(jnp.ones(2))
    meta = strux.metadata(saved)
    meta["static removed"] = "123"
    with pytest.raises(ValueError, match="at removed:.*absent"):
        strux.from_dict(
            strux.to_dict(saved), template=Module if use_class else saved, meta=meta
        )
