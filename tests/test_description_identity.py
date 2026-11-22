"""Compound checks distinguish descriptions whose display names coincide."""

import sys
import types

from beartype import beartype
from jaxtyping import TypeCheckError, jaxtyped
import pytest

import strux


def local_state():
    @strux.struct
    class State:
        value: int

    return State


@pytest.fixture(params=["repeated-local", "different-modules"])
def state_classes(request, monkeypatch):
    if request.param == "repeated-local":
        return local_state(), local_state()
    classes = []
    for name in ("identity_example_a", "identity_example_b"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        exec(
            "import strux\n@strux.struct\nclass State:\n    value: int\n", vars(module)
        )
        classes.append(module.State)
    return classes


@pytest.mark.parametrize("form", ["plain", "Struct", "Batch", "InstanceOf"])
def test_same_named_classes_have_independent_argument_and_return_checks(
    state_classes, form
):
    first, second = state_classes

    def make_annotation(cls):
        if form == "plain":
            return cls
        if form == "InstanceOf":
            return strux.InstanceOf[cls]
        return getattr(strux, form)[cls, ""]

    def make_checker(cls, result=None):
        annotation = tuple[make_annotation(cls), make_annotation(cls)]
        decorator = strux.checked if form == "plain" else jaxtyped(typechecker=beartype)

        @decorator
        def identity(value: annotation) -> annotation:
            return value if result is None else result

        return identity

    first_pair = (first(1), first(2))
    second_pair = (second(1), second(2))
    first_check = make_checker(first)
    second_check = make_checker(second)
    # Warm the first check before constructing the second backend's cache.
    assert first_check(first_pair) is first_pair
    assert second_check(second_pair) is second_pair
    with pytest.raises(TypeCheckError, match="parameters"):
        second_check(first_pair)
    with pytest.raises(TypeCheckError, match="parameters"):
        first_check(second_pair)
    with pytest.raises(TypeCheckError, match="return value"):
        make_checker(second, result=first_pair)(second_pair)


def test_distinct_transformations_with_identical_repr_remain_distinct():
    class ScalarType(type):
        def __repr__(cls):
            return "Scalar"

    # Replace's display text can coincide even for distinct annotation objects.
    first = ScalarType("Scalar", (int,), {})
    second = ScalarType("Scalar", (int,), {})
    state = local_state()
    first_hint = strux.Struct[state, strux.Replace(value=first)]
    second_hint = strux.Struct[state, strux.Replace(value=second)]
    assert first_hint.__name__ == second_hint.__name__

    def checker(hint):
        @jaxtyped(typechecker=beartype)
        def accept(value: tuple[hint]):
            return value

        return accept

    first_check, second_check = checker(first_hint), checker(second_hint)
    assert first_check((state(first(1)),))
    assert second_check((state(second(1)),))
    with pytest.raises(TypeCheckError):
        second_check((state(first(1)),))
