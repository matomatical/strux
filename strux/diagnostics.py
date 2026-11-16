"""Structured checking failures and their user-facing descriptions."""

import dataclasses

from strux.schema import ValidationError


@dataclasses.dataclass(frozen=True)
class _Axis:
    name: str
    key: str
    batch: bool = False


@dataclasses.dataclass(frozen=True)
class _Failure:
    path: str
    kind: str
    expected: object = None
    actual: object = None
    expected_shape: tuple | None = None
    actual_shape: tuple | None = None
    axis: _Axis | None = None
    alternatives: tuple = ()

    def render(self):
        if self.kind == "union":
            lines = [f"{self.path}: no alternative matches:"]
            for failure in self.alternatives:
                message = failure.render().replace("\n", "\n    ")
                lines.append(f"  - {message}")
            return "\n".join(lines)
        if self.kind == "dimension":
            label = "batch dimension" if self.axis.batch else "dimension"
            message = (
                f"{label} {self.axis.name!r} has size {self.actual}; "
                f"expected {self.expected}."
            )
        elif self.kind == "rank":
            message = f"expected rank {self.expected}; got rank {self.actual}."
        elif self.kind == "dtype":
            message = f"expected {self.expected} dtype; got {self.actual}."
        elif self.kind == "shape":
            message = "shape does not match the annotation."
        elif self.kind == "length":
            message = f"expected {self.expected} tuple entries; got {self.actual}."
        elif self.kind == "key":
            message = f"key {self.actual!r} does not match {self.expected}."
        else:
            message = f"expected {self.expected}; got {self.actual}."
        lines = [f"{self.path}: {message}"]
        if self.expected_shape is not None:
            dims = ", ".join(self.expected_shape)
            if len(self.expected_shape) == 1:
                dims += ","
            lines.append(f"  Expected shape: ({dims})")
            lines.append(f"  Actual shape:   {self.actual_shape}")
        return "\n".join(lines)

    def error(self):
        error = ValidationError(self.render())
        error._failure = self
        return error


def _union_failure(path, errors):
    failures = [error._failure for error in errors]
    # Discard incompatible root representations only when an alternative
    # reached their contents. Shape and dtype alternatives remain distinct.
    relevant = [
        failure
        for failure in failures
        if failure.path != path or failure.kind != "type"
    ]
    failures = relevant or failures
    unique = []
    for failure in failures:
        if failure not in unique:
            unique.append(failure)
    if len(unique) == 1:
        return unique[0].error()
    return _Failure(path, "union", alternatives=tuple(unique)).error()


def _leaf_failure(value, leaf, axes, path, bindings):
    """Explain a failed check; never decide whether a value is valid.

    Only positional ranks, fixed dimensions and plain named dimensions get
    specific explanations. Other shape syntax retains its written form in
    the fallback diagnostic. Bindings are the snapshot before the failed
    check, so failed alternatives cannot contribute constraints.
    """
    hint = getattr(leaf, "jaxtype", None)
    if hint is None:
        expected = leaf.cls.__name__
        if axes:
            expected = f"array with {expected} elements"
        return _Failure(path, "type", expected, type(value).__name__).error()
    if not isinstance(value, hint.array_type):
        return _Failure(
            path, "type", hint.array_type.__name__, type(value).__name__
        ).error()

    shape = tuple(value.shape)
    details = dict(expected_shape=tuple(axis.name for axis in axes), actual_shape=shape)
    # A shape-unconstrained check delegates dtype handling to jaxtyping.
    if not isinstance(value, hint.dtype[hint.array_type, "..."]):
        return _Failure(
            path, "dtype", hint.dtype.__name__, str(value.dtype), **details
        ).error()
    if any(axis.name == "..." or "*" in axis.name for axis in axes):
        return _Failure(path, "shape", **details).error()
    if len(axes) != len(shape):
        return _Failure(path, "rank", len(axes), len(shape), **details).error()

    sizes = dict(bindings)
    for axis, size in zip(axes, shape):
        if axis.name.isdigit():
            expected = int(axis.name)
        elif axis.name.isidentifier() and not axis.name.startswith("_"):
            expected = sizes.setdefault(axis.key, size)
        else:
            continue
        if expected != size:
            return _Failure(
                path, "dimension", expected, size, axis=axis, **details
            ).error()
    return _Failure(path, "shape", **details).error()
