"""Composable descriptions of struct fields and explicit runtime checking.

Struct descriptions retain ordinary value classes. Shape names declared by
an element are local to each occurrence of that element; batch dimensions
introduced by an outer description can connect checked arguments and returns.
"""

import contextvars
import dataclasses
import functools
import itertools
import re
import typing

import jax
import jaxtyping

from strux.diagnostics import _Axis, _Failure, _leaf_failure, _union_failure
from strux.schema import (
    SchemaError,
    ValidationError,
    schema,
    _parse_hint,
    _ArraySpec,
    _PyScalarSpec,
    _NoneSpec,
    _ClassSpec,
    _ContainerSpec,
    _UnionSpec,
)


@dataclasses.dataclass(frozen=True, init=False)
class Replace:
    """Replace named data-field annotations; preserve all other fields."""

    fields: tuple

    def __init__(self, **fields):
        object.__setattr__(self, "fields", tuple(fields.items()))

    def __repr__(self):
        return "Replace(" + ", ".join(f"{k}={v!r}" for k, v in self.fields) + ")"


@dataclasses.dataclass(frozen=True)
class Map:
    """Apply a pure annotation callback once to each current numerical leaf.

    A callback can introduce a subtree. The same Map does not revisit that
    subtree; subsequent maps descend into it. Static fields and None are
    not numerical leaves. Callbacks should not depend on invocation count.
    """

    function: typing.Callable

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("Map requires an annotation callable")


class InstanceOf:
    """Require class membership without checking a value's field annotations."""

    def __class_getitem__(cls, target):
        if not isinstance(target, type):
            raise TypeError("InstanceOf takes a Python class")
        return _make_instance_annotation(target)


class _InstanceAnnotationMeta(type):
    def __instancecheck__(cls, value):
        return isinstance(value, cls._instance_type)


@functools.lru_cache(maxsize=None)
def _make_instance_annotation(target):
    return _InstanceAnnotationMeta(
        f"InstanceOf[{target.__name__}]", (), {"_instance_type": target}
    )


class Batch:
    """Prepend scoped batch dimensions to a numerical pytree annotation.

    Batch[Class, dims] has the same meaning as Struct[Class, dims]. Array
    annotations and scalar shorthand are also supported. The second argument
    is always a dimension string, not a Map or Replace transformation.
    """

    def __new__(cls, annotation, dims):
        return cls[annotation, dims]

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Batch takes an annotation and a dimension string")
        annotation, dims = item
        if not isinstance(dims, str):
            raise TypeError("Batch dimensions must be a string")
        if isinstance(annotation, _BatchAnnotationMeta):
            dims = f"{dims} {annotation.dims}"
            annotation = annotation.annotation
        return _make_batch_annotation(annotation, " ".join(dims.split()))


class Struct:
    """Struct[Class, transformations...] describes values without wrapping them."""

    def __init__(self):
        raise TypeError("Struct is a type form, not a value constructor")

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError(
                'Struct takes a class and at least one transformation; use Struct[Class, ""] for an element'
            )
        target, *operations = item
        parts = _struct_parts(target)
        if parts is None:
            raise TypeError("Struct expects an annotated dataclass class")
        target, previous = parts
        return _make_struct_annotation(
            target, _normalise_operations((*previous, *operations))
        )


def _struct_parts(annotation):
    """Resolve a struct description without erasing its transformation order."""
    if isinstance(annotation, _StructAnnotationMeta):
        return annotation._struct_type, annotation._functors
    if isinstance(annotation, _BatchAnnotationMeta):
        parts = _struct_parts(annotation.annotation)
        if parts is None:
            return None
        cls, operations = parts
        return cls, _normalise_operations((*operations, annotation.dims))
    target = typing.get_origin(annotation) or annotation
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        return target, ()
    return None


def _normalise_operations(operations):
    normalised = []
    for operation in operations:
        if not isinstance(operation, (str, Replace, Map)):
            raise TypeError(
                "Struct transformations must be dims strings or functors (Replace or Map)"
            )
        if isinstance(operation, str):
            operation = " ".join(operation.split())
            if normalised and isinstance(normalised[-1], str):
                normalised[-1] = f"{operation} {normalised[-1]}".strip()
                continue
        normalised.append(operation)
    return tuple(normalised)


@functools.lru_cache(maxsize=None)
def _make_batch_annotation(annotation, dims):
    _parse_hint(annotation, "Batch element")
    parts = _struct_parts(annotation)
    if parts is not None:
        _fields(*parts)
    return _BatchAnnotationMeta(
        f"Batch[{getattr(annotation, '__name__', repr(annotation))}, {dims!r}]",
        (),
        {"annotation": annotation, "dims": dims},
    )


@functools.lru_cache(maxsize=None)
def _make_struct_annotation(cls, operations):
    # Check field names and annotation syntax when the description is declared.
    _fields(cls, operations)
    return _StructAnnotationMeta(
        f"Struct[{cls.__name__}, {', '.join(map(repr, operations))}]",
        (),
        {"_struct_type": cls, "_functors": operations},
    )


def _fields(cls, operations):
    fields = dict(schema(cls).fields)
    for operation in operations:
        if isinstance(operation, _Operation):
            operation = operation.operation
        if isinstance(operation, Replace):
            for name, hint in operation.fields:
                if name not in fields:
                    raise SchemaError(f"{cls.__name__}: unknown data field {name!r}")
                fields[name] = _parse_hint(hint, f"{cls.__name__}.{name}")
        else:
            fields = {
                name: _transform(spec, operation) for name, spec in fields.items()
            }
    return fields


@dataclasses.dataclass(frozen=True)
class _BatchSpec:
    element: object
    dims: str
    # None means dimension names have not yet been assigned a runtime scope.
    scope: object = None


@dataclasses.dataclass(frozen=True)
class _ScopedSpec:
    element: object
    scope: str


@dataclasses.dataclass(frozen=True)
class _Operation:
    operation: object
    scope: object = None


def _batch_spec(spec, dims, scope=None):
    """Retain the boundary between new batch axes and the element annotation."""
    return _BatchSpec(spec, dims, scope) if dims.strip() else spec


def _transform(spec, operation, batch_scope=None):
    """Apply one operation, deferring traversal through class references."""
    spec = _normalise_batch(spec)
    if isinstance(spec, _ClassSpec):
        if not spec.check_fields:
            if operation == "":
                return spec
            raise SchemaError(
                "InstanceOf exposes no leaf annotations to transform; "
                "use a class or Struct description instead"
            )
        return dataclasses.replace(
            spec, transforms=spec.transforms + (_Operation(operation, batch_scope),)
        )
    if isinstance(spec, _ContainerSpec):
        return dataclasses.replace(
            spec, elems=tuple(_transform(e, operation, batch_scope) for e in spec.elems)
        )
    if isinstance(spec, _UnionSpec):
        return _UnionSpec(
            tuple(_transform(e, operation, batch_scope) for e in spec.arms)
        )
    if isinstance(spec, _NoneSpec):
        return spec
    if isinstance(operation, str):
        return _batch_spec(spec, operation, batch_scope)
    if isinstance(spec, _ScopedSpec):
        result = _transform(spec.element, operation, batch_scope)
        return _scope_elements(result, spec.scope)
    if not isinstance(operation, Map):
        raise SchemaError(
            "Replace applies to a struct's fields, not an individual leaf"
        )
    return _map_result(spec, operation, batch_scope)


def _normalise_batch(spec):
    if not isinstance(spec, _BatchSpec):
        return spec
    element = _normalise_batch(spec.element)
    if isinstance(element, (_ClassSpec, _ContainerSpec, _UnionSpec, _NoneSpec)):
        return _transform(element, spec.dims, spec.scope)
    return dataclasses.replace(spec, element=element)


def _map_result(spec, operation, batch_scope):
    hint = _as_hint(spec)
    result = _call_map(hint, operation)
    transformed = _result_spec(
        result, source=spec, source_hint=hint, batch_scope=batch_scope
    )
    return _scope_elements(transformed, _element_scope(spec))


@functools.lru_cache(maxsize=None)
def _call_map(hint, operation):
    return operation.function(hint)


def _element_scope(spec):
    if isinstance(spec, _ScopedSpec):
        return spec.scope
    if isinstance(spec, _BatchSpec):
        return _element_scope(spec.element)
    return None


def _result_spec(hint, *, source=None, source_hint=None, batch_scope=None):
    if source is not None and hint == source_hint:
        return source
    if isinstance(hint, _BatchAnnotationMeta):
        return _batch_spec(
            _result_spec(
                hint.annotation,
                source=source,
                source_hint=source_hint,
                batch_scope=batch_scope,
            ),
            hint.dims,
        )
    if isinstance(hint, _StructAnnotationMeta) and batch_scope is not None:
        # A description introduced by a Map belongs to that transformation's
        # batch scope, just like an equivalent Batch returned by the callback.
        return _ClassSpec(
            hint._struct_type,
            transforms=tuple(_Operation(op, batch_scope) for op in hint._functors),
        )
    return _parse_hint(hint, "Map result")


def _as_hint(spec):
    if isinstance(spec, _ScopedSpec):
        return _as_hint(spec.element)
    if isinstance(spec, _ArraySpec):
        return spec.jaxtype
    if isinstance(spec, _PyScalarSpec):
        return spec.cls
    if isinstance(spec, _BatchSpec):
        return Batch[_as_hint(spec.element), spec.dims]
    raise SchemaError(f"cannot expose {spec!r} as a leaf annotation")


def _scope_elements(spec, scope):
    if scope is None:
        return spec
    if isinstance(spec, (_ArraySpec, _PyScalarSpec)):
        return _ScopedSpec(spec, scope)
    if isinstance(spec, _BatchSpec):
        return dataclasses.replace(spec, element=_scope_elements(spec.element, scope))
    if isinstance(spec, _UnionSpec):
        return _UnionSpec(tuple(_scope_elements(s, scope) for s in spec.arms))
    if isinstance(spec, _ContainerSpec):
        return dataclasses.replace(
            spec, elems=tuple(_scope_elements(s, scope) for s in spec.elems)
        )
    return spec


def _scope_batches(spec, scope):
    if isinstance(spec, _BatchSpec):
        return dataclasses.replace(
            spec,
            scope=spec.scope if spec.scope is not None else scope,
            element=_scope_batches(spec.element, scope),
        )
    if isinstance(spec, _ScopedSpec):
        return dataclasses.replace(spec, element=_scope_batches(spec.element, scope))
    if isinstance(spec, _UnionSpec):
        return _UnionSpec(tuple(_scope_batches(s, scope) for s in spec.arms))
    if isinstance(spec, _ContainerSpec):
        return dataclasses.replace(
            spec, elems=tuple(_scope_batches(s, scope) for s in spec.elems)
        )
    return spec


_scope_counter = contextvars.ContextVar("strux_scope_counter")
_EXTERNAL_SCOPE = "struxexternal_"


@dataclasses.dataclass(frozen=True)
class _BatchScope:
    prefix: str
    anonymous: str


def _qualify(dims, scope):
    if isinstance(scope, _BatchScope):
        tokens = []
        for index, token in enumerate(dims.split()):
            if token == "...":
                token = "*" + scope.anonymous + str(index)
            elif token == "_":
                token = scope.anonymous + str(index)
            else:
                token = _qualify(token, scope.prefix)
            tokens.append(token)
        return " ".join(tokens)
    if not scope:
        return dims

    def qualify_token(token):
        if token.lstrip("#*?").startswith("_"):
            return token
        parts = re.split(r"(\{[^}]*\})", token)
        for index in range(0, len(parts), 2):
            text = parts[index]
            parts[index] = re.sub(
                r"\b[A-Za-z_]\w*\b",
                lambda match: (
                    match[0]
                    if text[match.end() :].startswith("(")
                    else scope + match[0]
                ),
                text,
            )
        return "".join(parts)

    return " ".join(qualify_token(token) for token in dims.split())


def _leaf_hint(spec, scope):
    if isinstance(spec, _ScopedSpec):
        return _leaf_hint(spec.element, spec.scope)
    if isinstance(spec, _ArraySpec):
        hint = spec.jaxtype
        return hint.dtype[hint.array_type, _qualify(hint.dim_str, scope)]
    if isinstance(spec, _PyScalarSpec):
        return spec.cls
    if isinstance(spec, _BatchSpec):
        hint = _leaf_hint(spec.element, scope)
        if not hasattr(hint, "dim_str"):
            return None
        dims = _qualify(spec.dims, spec.scope if spec.scope is not None else scope)
        return hint.dtype[hint.array_type, f"{dims} {hint.dim_str}".strip()]
    raise SchemaError(f"not a leaf description: {spec!r}")


# The adapter isolates the private jaxtyping memo API needed to roll back
# failed union arms and publish only outer batch dimensions to @jaxtyped.
def _memo():
    from jaxtyping._storage import get_shape_memo

    return tuple(dict(part) for part in get_shape_memo())


def _set_memo(memo):
    from jaxtyping._storage import set_shape_memo

    set_shape_memo(*memo)


def _publish_memo(memo):
    from jaxtyping._storage import get_shape_memo

    # Jaxtyping's error formatter retains these dictionary objects. Publish
    # into them so both subsequent checks and diagnostics see the bindings.
    for current, updated in zip(get_shape_memo(), memo):
        current.clear()
        current.update(updated)


def _diagnostic_parts(spec, scope=""):
    if isinstance(spec, _ScopedSpec):
        return _diagnostic_parts(spec.element, spec.scope)
    if isinstance(spec, _BatchSpec):
        leaf, axes = _diagnostic_parts(spec.element, scope)
        batch_scope = spec.scope if spec.scope is not None else scope
        prefix = tuple(
            _Axis(name, key, batch=True)
            for name, key in zip(
                spec.dims.split(), _qualify(spec.dims, batch_scope).split()
            )
        )
        return leaf, prefix + axes
    if isinstance(spec, _ArraySpec):
        dims = spec.jaxtype.dim_str
        axes = tuple(
            _Axis(name, key)
            for name, key in zip(dims.split(), _qualify(dims, scope).split())
        )
        return spec, axes
    return spec, ()


def _explain_leaf(value, spec, path, initial):
    # Diagnostic probes must not change bindings, including while a union
    # generator is suspended or a partition search is being unwound.
    _set_memo(tuple(dict(part) for part in initial))
    try:
        leaf, axes = _diagnostic_parts(spec)
        return _leaf_failure(value, leaf, axes, path, initial[0])
    finally:
        _set_memo(initial)


def _matches(value, spec, path, inherited=()):
    spec = _normalise_batch(spec)
    if isinstance(spec, _ClassSpec):
        if not isinstance(value, spec.cls):
            raise _Failure(
                path, "type", spec.cls.__name__, type(value).__name__
            ).error()
        if not spec.check_fields:
            yield
            return
        if not dataclasses.is_dataclass(value):
            raise SchemaError(
                f"{path}: checking requires an annotated dataclass pytree"
            )
        local = f"struxlocal{next(_scope_counter.get())}_"
        fields = {
            k: _scope_elements(s, local) for k, s in schema(type(value)).fields.items()
        }
        own = (_Operation(" ".join(spec.dims)),) if spec.dims else ()
        operations = own + spec.transforms + inherited
        local_batch = None
        for bound in operations:
            operation = bound.operation
            scope = bound.scope
            if scope is None:
                if local_batch is None:
                    local_batch = f"struxderived{next(_scope_counter.get())}_"
                scope = local_batch
            if not isinstance(scope, _BatchScope):
                scope = _BatchScope(
                    scope, f"struxbatchlocal{next(_scope_counter.get())}_"
                )
            if isinstance(operation, Replace):
                for name, hint in operation.fields:
                    if name not in fields:
                        raise SchemaError(f"{path}: unknown data field {name!r}")
                    fields[name] = _scope_elements(
                        _parse_hint(hint, path + "." + name), local
                    )
            else:
                fields = {
                    k: _scope_batches(_transform(s, operation, scope), scope)
                    for k, s in fields.items()
                }
        yield from _sequence(
            [
                (getattr(value, name), field_spec, f"{path}.{name}")
                for name, field_spec in fields.items()
            ]
        )
    elif isinstance(spec, _ContainerSpec):
        expected = (
            dict if spec.kind == "dict" else list if spec.kind == "list" else tuple
        )
        if not isinstance(value, expected):
            raise _Failure(
                path, "type", expected.__name__, type(value).__name__
            ).error()
        if spec.kind == "tuple":
            if len(value) != len(spec.elems):
                raise _Failure(path, "length", len(spec.elems), len(value)).error()
            items = [(i, v, s) for i, (v, s) in enumerate(zip(value, spec.elems))]
        else:
            if spec.kind == "dict":
                for key in value:
                    if not _key_matches(key, spec.keytype):
                        raise _Failure(path, "key", repr(spec.keytype), key).error()
            entries = value.items() if spec.kind == "dict" else enumerate(value)
            items = [(k, v, spec.elems[0]) for k, v in entries]
        yield from _sequence(
            [(child, childspec, f"{path}[{key!r}]") for key, child, childspec in items]
        )
    elif isinstance(spec, _UnionSpec):
        initial = _memo()
        failures = []
        matched = False
        for arm in spec.arms:
            _set_memo(tuple(dict(p) for p in initial))
            try:
                for _ in _matches(value, arm, path):
                    matched = True
                    yield
            except ValidationError as error:
                failures.append(error)
        _set_memo(initial)
        if not matched:
            raise _union_failure(path, failures)
    elif isinstance(spec, _NoneSpec):
        if value is not None:
            raise _Failure(path, "type", "None", type(value).__name__).error()
        yield
    else:
        initial = _memo()
        try:
            hint = _leaf_hint(spec, "")
        except ValueError as error:
            if "variadic" not in str(error):
                raise
            yield from _partitioned_matches(value, spec, path)
            return
        if hint is None or not isinstance(value, hint):
            raise _explain_leaf(value, spec, path, initial)
        try:
            yield
        finally:
            _set_memo(initial)


def _key_matches(value, annotation):
    if annotation is typing.Any:
        return True
    if typing.get_origin(annotation) is tuple:
        arguments = typing.get_args(annotation)
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_key_matches(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _key_matches(item, hint) for item, hint in zip(value, arguments)
        )
    try:
        return isinstance(value, annotation)
    except TypeError as error:
        raise SchemaError(
            f"unsupported dictionary key annotation {annotation!r}"
        ) from error


def _sequence(items):
    """Backtrack over sibling constraints without recursing per list entry."""
    if not items:
        yield
        return
    stack = []
    depth = 0
    matched = False
    best_error = None
    failure_depth = -1
    try:
        while depth >= 0:
            if depth == len(items):
                matched = True
                yield
                depth -= 1
                continue
            if len(stack) == depth:
                stack.append(_matches(*items[depth]))
            try:
                next(stack[depth])
                depth += 1
            except (StopIteration, ValidationError) as error:
                if isinstance(error, ValidationError) and depth > failure_depth:
                    best_error = error
                    failure_depth = depth
                stack.pop().close()
                depth -= 1
        if not matched and best_error is not None:
            raise best_error
    finally:
        for choices in reversed(stack):
            choices.close()


def _shape_parts(spec, scope=""):
    if isinstance(spec, _ScopedSpec):
        return _shape_parts(spec.element, spec.scope)
    if isinstance(spec, _BatchSpec):
        hint, parts = _shape_parts(spec.element, scope)
        prefix = _qualify(spec.dims, spec.scope if spec.scope is not None else scope)
        return hint, [prefix, *parts]
    hint = spec.jaxtype
    return hint, [_qualify(hint.dim_str, scope)]


def _partitioned_matches(value, spec, path):
    """Match separate variadic batch/element segments with backtracking."""
    initial = _memo()
    hint, parts = _shape_parts(spec)
    if not isinstance(value, hint.dtype[hint.array_type, "..."]):
        raise _explain_leaf(value, spec, path, initial)

    def partitions(shape, remaining):
        if not remaining:
            if not shape:
                yield
            return
        dims, *rest = remaining
        tokens = dims.split()
        variadic = any(token == "..." or token.startswith("*") for token in tokens)
        minimum = len(tokens) - int(variadic)
        lengths = range(minimum, len(shape) + 1) if variadic else (minimum,)
        for length in lengths:
            if length > len(shape):
                continue
            initial = _memo()
            try:
                prefix = jax.ShapeDtypeStruct(shape[:length], value.dtype)
                if isinstance(prefix, jaxtyping.Shaped[jax.ShapeDtypeStruct, dims]):
                    yield from partitions(shape[length:], rest)
            finally:
                _set_memo(initial)

    matched = False
    for _ in partitions(tuple(value.shape), parts):
        matched = True
        yield
    if not matched:
        raise _explain_leaf(value, spec, path, initial)


def _check_description(value, annotation, *, publish):
    external = _memo() if publish else ({}, {}, {}, {})
    # Separate public dimension names from implementation-local names. This
    # also permits users to choose names resembling our internal prefixes.
    working = (
        {_EXTERNAL_SCOPE + name: size for name, size in external[0].items()},
        {_EXTERNAL_SCOPE + name: size for name, size in external[1].items()},
        dict(external[2]),
        dict(external[3]),
    )
    counter_token = _scope_counter.set(itertools.count())
    try:
        with jaxtyping.jaxtyped("context"):
            _set_memo(working)
            parts = _struct_parts(annotation)
            if parts is not None:
                cls, transformations = parts
                spec = _ClassSpec(cls)
                operations = tuple(
                    _Operation(op, _EXTERNAL_SCOPE) for op in transformations
                )
                matches = _matches(value, spec, cls.__name__, operations)
            else:
                local = f"struxlocal{next(_scope_counter.get())}_"
                batch_scope = _BatchScope(
                    _EXTERNAL_SCOPE,
                    f"struxbatchlocal{next(_scope_counter.get())}_",
                )
                spec = _scope_elements(_result_spec(annotation), local)
                spec = _scope_batches(spec, batch_scope)
                matches = _matches(value, spec, annotation.__name__)
            try:
                next(matches)
                result = _memo()
            except StopIteration:
                raise ValidationError(
                    f"{annotation.__name__}: fields do not match"
                ) from None
            finally:
                matches.close()
    finally:
        _scope_counter.reset(counter_token)
    if publish:
        exported = tuple(
            {
                name.removeprefix(_EXTERNAL_SCOPE): size
                for name, size in part.items()
                if name.startswith(_EXTERNAL_SCOPE)
            }
            for part in result[:2]
        )
        _publish_memo((*exported, result[2], result[3]))


def check(value, annotation=None):
    """Check a value without changing it. Raise ValidationError on mismatch.

    With no annotation, require the declared element fields exactly.
    Explicit Struct and Batch descriptions use a fresh dimension scope and
    return None. A Batch description can also check an array or other
    supported numerical pytree directly. InstanceOf checks class membership
    without inspecting fields.
    """
    if isinstance(annotation, _InstanceAnnotationMeta):
        if not isinstance(value, annotation):
            raise _Failure(
                annotation.__name__,
                "type",
                annotation._instance_type.__name__,
                type(value).__name__,
            ).error()
        return
    if annotation is None:
        annotation = Struct[type(value), ""]
    elif isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        annotation = Struct[annotation, ""]
    if not isinstance(annotation, _DescriptionMeta):
        raise TypeError(
            "check expects a Struct, Batch, or InstanceOf description or dataclass class"
        )
    _check_description(value, annotation, publish=False)


class _DescriptionMeta(type):
    def __instancecheck__(cls, value):
        try:
            _check_description(value, cls, publish=True)
        except ValidationError:
            return False
        return True


class _StructAnnotationMeta(_DescriptionMeta):
    pass


class _BatchAnnotationMeta(_DescriptionMeta):
    pass
