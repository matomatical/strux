"""Function-boundary checking with recursive strux class annotations."""

import dataclasses
import functools
import inspect
import types
import typing

import jaxtyping

from strux.annotate import Struct, _DescriptionMeta, _InstanceAnnotationMeta

_DEFAULT = object()
_Function = typing.TypeVar("_Function", bound=typing.Callable)


@typing.overload
def checked(fn: _Function, *, typechecker=...) -> _Function: ...


@typing.overload
def checked(*, typechecker=...) -> typing.Callable[[_Function], _Function]: ...


def checked(fn=None, *, typechecker=_DEFAULT):
    """Check function arguments and returns against their annotations.

    Plain strux classes request their declared fields recursively. Struct
    and Batch descriptions keep their explicit transformations; InstanceOf
    requests only class membership. Constructors and replace are unaffected.

    Beartype is the default backend, installed with ``strux[beartype]``.
    Supply another jaxtyping-compatible decorator with ``typechecker=...``,
    or None to establish a jaxtyping context without automatic checks.

    Annotation resolution and checker construction happen on the first call
    and are cached (per receiver class for methods). This permits references
    to the enclosing class and Self. Unresolved annotations raise an error;
    they are never silently treated as Any. Original annotations are retained.

    Supports synchronous functions, methods, and property accessors. Place
    jax.jit outside checked so checks run during tracing. Async functions and
    generators are not supported.
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
            namespace = dict(localns)
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


def _translate(hint, receiver, aliases=()):
    """Translate instance annotations while retaining class objects and metadata."""

    def translate(child):
        return _translate(child, receiver, aliases)

    if hint is typing.Self:
        if receiver is None:
            raise TypeError(
                "Self in strux.checked requires an instance or class method"
            )
        return translate(receiver)
    if isinstance(hint, (_DescriptionMeta, _InstanceAnnotationMeta)):
        return hint
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
        return _translate(expanded, receiver, (*aliases, alias))
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
    return _map_arguments(hint, translate)


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
