"""Function-boundary checking with recursive strux class annotations."""

import collections.abc
import dataclasses
import functools
import inspect
import types
import typing

import jaxtyping

from strux.annotate import Struct, _DescriptionMeta, _InstanceAnnotationMeta
from strux.schema import _scalar_annotation

_DEFAULT = object()
_Function = typing.TypeVar("_Function", bound=typing.Callable)

# These arguments describe values, rather than a user-defined generic's
# nominal type parameters. Class-object annotations are handled separately.
_VALUE_ARGUMENT_ORIGINS = {
    typing.Annotated,
    typing.Union,
    types.UnionType,
    list,
    tuple,
    dict,
    set,
    frozenset,
    collections.abc.Callable,
    collections.abc.Collection,
    collections.abc.Iterable,
    collections.abc.Iterator,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Mapping,
    collections.abc.MutableMapping,
    collections.abc.Set,
    collections.abc.MutableSet,
}


@typing.overload
def checked(fn: _Function, *, typechecker=...) -> _Function: ...


@typing.overload
def checked(*, typechecker=...) -> typing.Callable[[_Function], _Function]: ...


def checked(fn=None, *, typechecker=_DEFAULT):
    """Check function arguments and returns against their annotations.

    Plain strux classes request their declared fields recursively. Struct
    and Batch descriptions keep their explicit transformations; InstanceOf
    requests only class membership. Constructors and replace are unaffected.
    Scalar value annotations bool, int, float and complex accept matching
    Python and NumPy scalars or zero-dimensional arrays, including JAX tracers,
    using the same scalar convention as struct fields, including inside unions
    and containers. InstanceOf requests a strict Python instance check.
    Class objects and nominal generic type arguments
    retain their scalar class annotations. Scalar branches introduced by
    jaxtyping ArrayLike annotations expand too; use concrete array types to
    require a particular dtype.

    Beartype is the default backend, installed with ``strux[beartype]``.
    Supply another jaxtyping-compatible decorator with ``typechecker=...``,
    or None to establish a jaxtyping context without automatic checks.

    Annotation resolution and checker construction happen on the first call
    and are cached (per receiver class for methods). This permits references
    to the enclosing class and Self. Unresolved annotations raise an error;
    they are never silently treated as Any. Original annotations are retained.

    Supports synchronous functions, methods, and property accessors. Place
    jax.jit outside checked so checks run during tracing. Async functions and
    generators are not supported. Applying checked again to its own wrapper
    replaces its checking configuration instead of nesting checking contexts.
    This decorator owns its jaxtyping context; do not supply it as the
    typechecker of another jaxtyped wrapper.
    """
    if fn is None:
        return functools.partial(checked, typechecker=typechecker)
    frame = inspect.currentframe()
    try:
        localns = dict(frame.f_back.f_locals)
    finally:
        del frame
    if typechecker is _DEFAULT:
        try:
            from beartype import beartype
        except ImportError as error:
            raise ImportError(
                "strux.checked requires the optional beartype dependency; "
                "install 'strux[beartype]' or pass typechecker explicitly"
            ) from error
        typechecker = beartype
    return _decorate(fn, typechecker, localns)


def _decorate(fn, typechecker, localns):
    if isinstance(fn, (classmethod, staticmethod)):
        return type(fn)(_decorate(fn.__func__, typechecker, localns))
    if isinstance(fn, property):
        return property(
            *(
                _decorate(accessor, typechecker, localns) if accessor else None
                for accessor in (fn.fget, fn.fset, fn.fdel)
            ),
            doc=fn.__doc__,
        )
    if not inspect.isfunction(fn):
        raise TypeError("strux.checked decorates functions, not classes or values")
    no_type_check = getattr(fn, "__no_type_check__", False)
    if getattr(fn, "_strux_checked_code", None) is fn.__code__:
        localns = fn._strux_checked_localns
        fn = fn.__wrapped__
    if (
        inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
        or inspect.isasyncgenfunction(fn)
    ):
        raise TypeError("strux.checked supports synchronous, non-generator functions")
    if typechecker is not None and not callable(typechecker):
        raise TypeError("typechecker must be a function decorator or None")

    cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if (
            jaxtyping.config.jaxtyping_disable
            or getattr(fn, "__no_type_check__", False)
            or getattr(wrapper, "__no_type_check__", False)
        ):
            return fn(*args, **kwargs)
        receiver = _receiver_class(fn, args, kwargs)
        if receiver not in cache:
            # Class type parameters occupy an annotation scope outside the
            # class dictionary. get_type_hints(function) only supplies the
            # function's own type parameters, so retain the enclosing ones.
            namespace = {
                param.__name__: param for param in localns.get("__type_params__", ())
            }
            namespace.update(localns)
            if receiver is not None:
                namespace.update(
                    {cls.__name__: cls for cls in reversed(receiver.__mro__)}
                )
            hints = {}
            if typechecker is not None:
                try:
                    hints = typing.get_type_hints(
                        fn,
                        globalns=fn.__globals__,
                        localns=namespace,
                        include_extras=True,
                    )
                except NameError as error:
                    raise TypeError(
                        f"{fn.__qualname__}: strux.checked could not resolve annotations: "
                        f"{error}"
                    ) from error
            translated = {
                name: _translate(hint, receiver) for name, hint in hints.items()
            }
            # String format avoids evaluating deferred annotations a second time
            # without the local namespace used by get_type_hints above.
            if hasattr(inspect, "Format"):
                signature = inspect.signature(
                    fn, annotation_format=inspect.Format.STRING
                )
            else:
                signature = inspect.signature(fn, eval_str=False)
            signature = signature.replace(
                parameters=[
                    param.replace(
                        annotation=translated.get(name, inspect.Parameter.empty)
                    )
                    for name, param in signature.parameters.items()
                ],
                return_annotation=translated.get("return", inspect.Signature.empty),
            )

            @functools.wraps(fn)
            def proxy(*args, **kwargs):
                return fn(*args, **kwargs)

            proxy.__annotations__ = translated
            proxy.__signature__ = signature
            cache[receiver] = jaxtyping.jaxtyped(proxy, typechecker=typechecker)
        return cache[receiver](*args, **kwargs)

    # Code identity distinguishes this wrapper from other decorators which
    # copy its attributes with functools.wraps.
    wrapper._strux_checked_code = wrapper.__code__
    wrapper._strux_checked_localns = localns
    if no_type_check:
        wrapper.__no_type_check__ = True
    return wrapper


def _receiver_class(fn, args, kwargs):
    """Identify a method receiver without interpreting its field annotations."""
    owner_name, _, _ = fn.__qualname__.rpartition(".")
    if not owner_name or owner_name.endswith("<locals>") or not fn.__code__.co_argcount:
        return None
    value = args[0] if args else kwargs.get(fn.__code__.co_varnames[0])
    cls = value if isinstance(value, type) else type(value)
    if any(base.__qualname__ == owner_name for base in cls.__mro__):
        return cls
    return None


def _translate(hint, receiver, aliases=(), *, scalar_values=True):
    """Translate instance annotations while retaining class objects and metadata."""

    def translate(child):
        return _translate(child, receiver, aliases, scalar_values=scalar_values)

    if hint is typing.Self:
        if receiver is None:
            raise TypeError(
                "Self in strux.checked requires an instance or class method"
            )
        return translate(receiver)
    if isinstance(hint, (_DescriptionMeta, _InstanceAnnotationMeta)):
        return hint
    if scalar_values and hint in (bool, int, float, complex):
        return _scalar_annotation(hint)
    origin = typing.get_origin(hint)
    alias = origin or hint
    if isinstance(alias, typing.TypeAliasType):
        if alias in aliases:
            raise TypeError("strux.checked does not support recursive type aliases")
        substitutions = dict(zip(alias.__type_params__, typing.get_args(hint)))

        def substitute(child):
            if isinstance(child, typing.TypeVar):
                return substitutions.get(child, child)
            return _map_arguments(child, substitute)

        expanded = substitute(alias.__value__)
        return _translate(
            expanded, receiver, (*aliases, alias), scalar_values=scalar_values
        )
    if isinstance(hint, typing.TypeVar):
        bound = hint.__bound__
        constraints = hint.__constraints__
        new_bound = translate(bound) if bound is not None else None
        new_constraints = tuple(translate(child) for child in constraints)
        if new_bound == bound and new_constraints == constraints:
            return hint
        return typing.TypeVar(
            hint.__name__,
            *new_constraints,
            bound=new_bound,
            covariant=hint.__covariant__,
            contravariant=hint.__contravariant__,
            infer_variance=hint.__infer_variance__,
        )
    if isinstance(hint, typing.ForwardRef):
        raise TypeError(f"strux.checked cannot check an unresolved annotation: {hint}")
    if origin is type:
        (target,) = typing.get_args(hint)
        if target is typing.Self:
            if receiver is None:
                raise TypeError("type[Self] requires an instance or class method")
            return type[receiver]
        return hint
    target = origin or hint
    if (
        isinstance(target, type)
        and dataclasses.is_dataclass(target)
        and hasattr(target, "_data_fields")
    ):
        return Struct[hint, ""]
    return _map_arguments(
        hint,
        lambda child: _translate(
            child,
            receiver,
            aliases,
            scalar_values=scalar_values and origin in _VALUE_ARGUMENT_ORIGINS,
        ),
    )


def _map_arguments(hint, transform):
    """Visit annotation arguments, leaving Literal values and metadata intact."""
    origin = typing.get_origin(hint)
    if origin is typing.Annotated:
        target, *metadata = typing.get_args(hint)
        return typing.Annotated[transform(target), *metadata]
    if origin is typing.Literal or origin is None:
        return hint
    args = typing.get_args(hint)
    transformed = tuple(
        [transform(item) for item in arg] if isinstance(arg, list) else transform(arg)
        for arg in args
    )
    if transformed == args:
        return hint
    if origin in (typing.Union, types.UnionType):
        return typing.Union[transformed]
    if hasattr(hint, "copy_with"):
        return hint.copy_with(transformed)
    return origin[transformed]
