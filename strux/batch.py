"""Infer shared leading batch shapes from default element annotations.

Inference uses structure and shape, not dtype validation. Explicit batch
hints can supply the indexing interpretation for transformed values.
"""

import dataclasses
import numbers

import jax

from strux.schema import (
    schema,
    ValidationError,
    _ArraySpec,
    _PyScalarSpec,
    _NoneSpec,
    _ClassSpec,
    _ContainerSpec,
    _UnionSpec,
)

# The candidate-set lattice: each (value, spec) pair yields the set of
# leading batch shapes B consistent with it — a finite frozenset of tuples,
# or _TOP (represented as None) meaning "unconstrained". Only pytree
# *structure* is unconstrained (None arms, empty containers): every leaf
# constrains the batch, and in particular a python scalar admits exactly
# batch () — batching produces arrays, never scalars, and strux does not
# broadcast. Shape queries require a unique shared prefix.
_TOP = None


def _solved_candidates(tree):
    """Compute candidates from current fields, including mutable containers."""
    cls = type(tree)
    meet = _BatchMeet()
    for name, spec in schema(cls).fields.items():
        path = f"{cls.__name__}.{name}"
        meet.add(path, _candidates(getattr(tree, name), spec, path))
    return meet.candidates


class _BatchMeet:
    """Accumulate per-field candidate sets, intersecting as we go."""

    def __init__(self):
        self.candidates = _TOP
        self.witnesses = []

    def add(self, path, candidates):
        if candidates is _TOP:
            return
        if self.candidates is _TOP:
            self.candidates = candidates
        else:
            met = self.candidates & candidates
            if not met:
                raise ValidationError(
                    "inconsistent batch shapes: "
                    f"{', '.join(self.witnesses)} admit batch shapes "
                    f"{_format_candidates(self.candidates)}, but {path} "
                    f"admits {_format_candidates(candidates)}"
                )
            self.candidates = met
        self.witnesses.append(path)


def _format_candidates(candidates):
    ordered = sorted(candidates, key=lambda b: (len(b), b))
    inner = ", ".join(str(b) for b in ordered[:4])
    if len(ordered) > 4:
        inner += ", ..."
    return "{" + inner + "}"


def _is_arraylike_value(value):
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _candidates(value, spec, path):
    """
    The set of batch shapes consistent with this value under this spec
    (frozenset of tuples, or _TOP for "unconstrained"). Raises
    ValidationError if no batch shape is consistent.
    """
    from strux.annotate import _BatchSpec

    if isinstance(spec, _BatchSpec):
        return _discount_dims(
            _candidates(value, spec.element, path),
            _ClassSpec(object, dims=tuple(spec.dims.split())),
            path,
        )
    if isinstance(spec, _ArraySpec):
        if not _is_arraylike_value(value):
            raise ValidationError(
                f"{path}: expected an array ({spec}), got " f"{type(value).__name__}"
            )
        shape = tuple(value.shape)
        if len(shape) < spec.min_ndim:
            raise ValidationError(
                f"{path}: expected at least {spec.min_ndim} element dims "
                f"({spec}), got shape {shape}"
            )
        for offset, size in spec.fixed:
            if shape[len(shape) + offset] != size:
                raise ValidationError(
                    f"{path}: expected element dims matching {spec}, got "
                    f"shape {shape} (trailing dim {offset} should be "
                    f"{size})"
                )
        if spec.ndim is None:
            return frozenset(shape[:k] for k in range(len(shape) - spec.min_ndim + 1))
        return frozenset((shape[: len(shape) - spec.ndim],))
    elif isinstance(spec, _PyScalarSpec):
        if isinstance(value, numbers.Number) or getattr(value, "shape", None) == ():
            # a python scalar is one element's worth of data: batch ()
            # exactly (batched structs carry arrays; strux never
            # broadcasts a scalar across a batch)
            return frozenset(((),))
        raise ValidationError(
            f"{path}: expected {spec.cls.__name__}, got " f"{type(value).__name__}"
        )
    elif isinstance(spec, _NoneSpec):
        if value is None:
            return _TOP
        raise ValidationError(f"{path}: expected None, got {type(value).__name__}")
    elif isinstance(spec, _ClassSpec):
        if not isinstance(value, spec.cls):
            raise ValidationError(
                f"{path}: expected an instance of {spec.cls.__name__}, got "
                f"{type(value).__name__}"
            )
        value_cls = type(value)
        if dataclasses.is_dataclass(value_cls):
            if spec.transforms:
                from strux.annotate import _fields

                fields = _fields(value_cls, spec.transforms)
                meet = _BatchMeet()
                for name, childspec in fields.items():
                    meet.add(
                        path + "." + name,
                        _candidates(getattr(value, name), childspec, path + "." + name),
                    )
                return _discount_dims(meet.candidates, spec, path)
            return _discount_dims(_solved_candidates(value), spec, path)
        # any other pytree: every array leaf constrains the batch as a
        # prefix of its shape; a scalar leaf admits batch () only
        meet = _BatchMeet()
        for keypath, leaf in jax.tree.flatten_with_path(value)[0]:
            leafpath = path + "".join(str(k) for k in keypath)
            if _is_arraylike_value(leaf):
                shape = tuple(leaf.shape)
                meet.add(
                    leafpath,
                    frozenset(shape[:k] for k in range(len(shape) + 1)),
                )
            elif isinstance(leaf, (bool, int, float, complex)):
                meet.add(leafpath, frozenset(((),)))
            else:
                raise ValidationError(
                    f"{leafpath}: value of type {type(leaf).__name__} "
                    "inside a data field is not an array; restructure, or "
                    "mark the field static"
                )
        return _discount_dims(meet.candidates, spec, path)
    elif isinstance(spec, _ContainerSpec):
        expected_cls = (
            dict if spec.kind == "dict" else (list if spec.kind == "list" else tuple)
        )
        if not isinstance(value, expected_cls):
            raise ValidationError(
                f"{path}: expected {spec}, got {type(value).__name__}"
            )
        if spec.kind == "dict":
            children = [(f"{path}[{k!r}]", v, spec.elems[0]) for k, v in value.items()]
        elif spec.kind in ("list", "tuple_variadic"):
            children = [(f"{path}[{i}]", v, spec.elems[0]) for i, v in enumerate(value)]
        else:
            if len(value) != len(spec.elems):
                raise ValidationError(
                    f"{path}: expected {spec} (length {len(spec.elems)}), "
                    f"got length {len(value)}"
                )
            children = [
                (f"{path}[{i}]", v, e)
                for i, (v, e) in enumerate(zip(value, spec.elems))
            ]
        meet = _BatchMeet()
        for childpath, child, childspec in children:
            meet.add(childpath, _candidates(child, childspec, childpath))
        return meet.candidates
    elif isinstance(spec, _UnionSpec):
        arm_candidates = []
        arm_failures = []
        for arm in spec.arms:
            try:
                arm_candidates.append(_candidates(value, arm, path))
            except ValidationError as e:
                arm_failures.append(f"as {arm}: {e}")
        if not arm_candidates:
            raise ValidationError(
                f"{path}: value matches no arm of {spec}: " + "; ".join(arm_failures)
            )
        if any(c is _TOP for c in arm_candidates):
            return _TOP
        return frozenset().union(*arm_candidates)
    else:
        raise AssertionError(f"unknown spec {spec!r}")


def _discount_dims(candidates, spec, path):
    """
    Discount the dims a `Struct[Cls, dims]` field annotation declares.

    A stacked field's leaves carry `enclosing_batch ++ declared_dims ++
    element_dims`, so the child's own solution is `enclosing_batch ++
    declared_dims` and the enclosing struct's contribution is what remains
    once the declared dims are taken off the trailing end.

    Raises ValidationError if the declared dims fit no solution the child
    admits. Dim tokens this cannot reason about (jaxtyping forms beyond
    integer literals, plain names, and one variadic) yield _TOP: the field
    simply constrains nothing, rather than constraining it wrongly.
    """
    tokens = spec.dims
    if not tokens or candidates is _TOP:
        return candidates
    for token in tokens:
        if token != "..." and not (token.isdigit() or token.isidentifier()):
            return _TOP
    if tokens.count("...") > 1:
        return _TOP
    fixed_width = len([t for t in tokens if t != "..."])
    variadic = "..." in tokens
    residuals = set()
    for shape in candidates:
        widths = range(fixed_width, len(shape) + 1) if variadic else (fixed_width,)
        for width in widths:
            if width > len(shape):
                continue
            cut = len(shape) - width
            if _match_batch(tokens, shape[cut:]):
                residuals.add(shape[:cut])
    if not residuals:
        raise ValidationError(
            f"{path}: expected element dims matching {spec}, but the value "
            f"admits batch shapes {_format_candidates(candidates)}"
        )
    return frozenset(residuals)


def _match_batch(tokens, shape):
    """
    Does this batch shape match the dims pattern? Supports integer
    literals, symbolic names (bound consistently within the pattern), and
    one "..." variadic. Patterns using other jaxtyping token forms are not
    refuted here (per-field checks remain authoritative for them).
    """
    for token in tokens:
        if token != "..." and not (token.isdigit() or token.isidentifier()):
            return True
    if tokens.count("...") > 1:
        return True
    if "..." in tokens:
        i = tokens.index("...")
        head, tail = tokens[:i], tokens[i + 1 :]
        if len(head) + len(tail) > len(shape):
            return False
        pairs = list(zip(head, shape)) + list(zip(reversed(tail), reversed(shape)))
    else:
        if len(tokens) != len(shape):
            return False
        pairs = list(zip(tokens, shape))
    bound = {}
    for token, size in pairs:
        if token.isdigit():
            if int(token) != size:
                return False
        elif bound.setdefault(token, size) != size:
            return False
    return True


def _bind_dims(value, spec, batch, bindings, path):
    """
    Bind the symbolic dim names in a spec to their sizes in this value,
    raising ValidationError if a name binds two different sizes. Names are
    scoped to the class whose annotations mention them: nested dataclasses
    are not entered (their names belong to their own occurrence).
    """
    if isinstance(spec, _ArraySpec):
        if not spec.names or not _is_arraylike_value(value):
            return
        shape = tuple(value.shape)
        for offset, dim_name in spec.names:
            size = shape[len(shape) + offset]
            if dim_name in bindings:
                previous_size, previous_path = bindings[dim_name]
                if previous_size != size:
                    raise ValidationError(
                        f"inconsistent dim '{dim_name}': {previous_size} at "
                        f"{previous_path}, but {size} at {path}"
                    )
            else:
                bindings[dim_name] = (size, path)
    elif isinstance(spec, _ContainerSpec):
        if isinstance(value, dict):
            for k, v in value.items():
                _bind_dims(v, spec.elems[0], batch, bindings, f"{path}[{k!r}]")
        elif isinstance(value, (list, tuple)):
            if spec.kind == "tuple" and len(value) == len(spec.elems):
                elem_specs = spec.elems
            else:
                elem_specs = (spec.elems[0],) * len(value)
            for i, (v, e) in enumerate(zip(value, elem_specs)):
                _bind_dims(v, e, batch, bindings, f"{path}[{i}]")
    elif isinstance(spec, _UnionSpec):
        # bind through the arm this value inhabits, when determinable
        matching = []
        for arm in spec.arms:
            try:
                candidates = _candidates(value, arm, path)
            except ValidationError:
                continue
            if candidates is _TOP or batch is None or batch in candidates:
                matching.append(arm)
        if len(matching) == 1:
            _bind_dims(value, matching[0], batch, bindings, path)
    # scalar and None specs carry no named dims; nested dataclasses and
    # other class-typed fields are their own scope


def _bind_names(instance, sch, batch):
    """
    Bind and check the symbolic dim names across a struct's fields (one
    namespace per class). `batch` is the solved batch shape when unique
    (used only to disambiguate union arms), else None.
    """
    bindings = {}
    cls_name = sch.cls.__name__
    for name, spec in sch.fields.items():
        _bind_dims(
            getattr(instance, name),
            spec,
            batch,
            bindings,
            f"{cls_name}.{name}",
        )
    return bindings
