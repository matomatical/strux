"""
The @struct decorator: transform an annotated class into a frozen dataclass
registered as a JAX pytree, with schema-checked construction and the
convenience members grafted on.
"""

import dataclasses
import functools
import inspect
import typing
import warnings

import jax

from strux.batch import (
    _CACHED_SOLUTION,
    _UNSOLVED,
    _is_arraylike_value,
    _validate_struct,
)
from strux.pprint import to_str, tree_format
from strux.serial import load, save
from strux.shapes import tree_getitem, tree_iter, tree_len, tree_shape, tree_size


# dataclass_transform (PEP 681) tells static type checkers like mypy that
# @strux.struct generates dataclass semantics (an __init__ from the field
# annotations, frozen instances)
@typing.dataclass_transform(
    frozen_default=True,
    field_specifiers=(dataclasses.field,),
)
def struct(
    Class=None,
    *,
    static_fieldnames: typing.Sequence[str] = (),
    check: bool = True,
):
    """
    Transform a class into an immutable dataclass that is also registered as a
    JAX PyTree. Can be used as a bare decorator or with keyword arguments:

    ```
    @strux.struct
    class MyDataClass:
        field1: int
        # etc.

    @strux.struct(static_fieldnames=("label",))
    class MyOtherDataClass:
        field1: int
        label: str
    ```

    By default, construction validates the data fields against the schema
    derived from the field annotations (see `strux.schema`): dtype kind and
    trailing (element) dims are enforced, leading batch dims are free but
    must agree across fields. Pass `check=False` to skip this validation
    (e.g. for a class constructed in a measured hot path).
    """
    if Class is None:
        return functools.partial(
            struct,
            static_fieldnames=static_fieldnames,
            check=check,
        )
    # wrap class as an immutable Python dataclass
    Dataclass = dataclasses.dataclass(Class, frozen=True)

    # decide which fields are data vs. static
    fields = [field.name for field in dataclasses.fields(Dataclass)]
    data_fields = [name for name in fields if name not in static_fieldnames]
    meta_fields = [name for name in fields if name in static_fieldnames]
    missing_fields = set(static_fieldnames) - set(meta_fields)
    if missing_fields:
        raise ValueError(f"Invalid static_fieldnames {missing_fields}")
    Dataclass._data_fields = data_fields
    Dataclass._meta_fields = meta_fields

    # register as a JAX pytree node. Reconstruction through JAX's tree
    # machinery (tree.map results, vmap/scan/jit outputs, gradients) is
    # *structural*: it bypasses __init__ and hence validation, because
    # transformed trees legitimately carry leaves that differ from the
    # declared element types (tree.map(jnp.array_equal, ...) yields bool
    # scalars, cotangents carry float0, masks change dtypes). Validation
    # guards the user API boundary instead: direct construction and
    # .replace. (For the same reason, __post_init__ logic runs on direct
    # construction only, never on tree reconstruction.)
    data_fields_tuple = tuple(data_fields)
    meta_fields_tuple = tuple(meta_fields)
    def _flatten_with_keys(obj):
        return (
            [
                (jax.tree_util.GetAttrKey(name), getattr(obj, name))
                for name in data_fields_tuple
            ],
            tuple(getattr(obj, name) for name in meta_fields_tuple),
        )
    def _flatten(obj):
        return (
            tuple(getattr(obj, name) for name in data_fields_tuple),
            tuple(getattr(obj, name) for name in meta_fields_tuple),
        )
    def _unflatten(meta, data):
        obj = object.__new__(Dataclass)
        for name, value in zip(data_fields_tuple, data):
            object.__setattr__(obj, name, value)
        for name, value in zip(meta_fields_tuple, meta):
            object.__setattr__(obj, name, value)
        return obj
    jax.tree_util.register_pytree_with_keys(
        Dataclass,
        _flatten_with_keys,
        _unflatten,
        flatten_func=_flatten,
    )

    # overwrite string render methods to use pretty printing
    if "__str__" not in Class.__dict__:
        Dataclass.__str__ = to_str
    if "__format__" not in Class.__dict__:
        Dataclass.__format__ = tree_format

    # add some other convenience methods
    if "replace" not in fields:
        # a replacement value with exactly the leaf layout (structure,
        # shapes, dtypes, python scalar types) of the field it replaces
        # cannot change the instance's validity, so it skips revalidation
        # and constructs structurally; anything else goes through the
        # constructor (and hence validation) as a direct construction
        # would. Classes with __post_init__ always take the constructor
        # path, so their hook always runs.
        if check and not hasattr(Class, "__post_init__"):
            Dataclass.replace = _checked_replace
        else:
            Dataclass.replace = dataclasses.replace
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'replace', so the "
            f"convenience method .replace() will not be available; use "
            f"dataclasses.replace(obj, ...) instead",
        )
    if "size" not in fields:
        Dataclass.size = property(tree_size)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'size', so the "
            f"convenience property .size will not be available; use "
            f"strux.tree_size(obj) instead",
        )
    if "shape" not in fields:
        Dataclass.shape = property(tree_shape)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'shape', so the "
            f"convenience property .shape will not be available; use "
            f"strux.tree_shape(obj) instead",
        )
    for dunder, method, module_name in (
        ("__getitem__", tree_getitem, "strux.tree_getitem"),
        ("__len__", tree_len, "strux.tree_len"),
        ("__iter__", tree_iter, "strux.tree_iter"),
    ):
        if dunder not in Class.__dict__:
            setattr(Dataclass, dunder, method)
        else:
            warnings.warn(
                f"{Class.__name__} defines {dunder}, so strux's batch "
                f"version will not be installed; the module-level "
                f"equivalent ({module_name}) remains available",
            )
    if "save" not in fields:
        def _save_method(self, path, *, fmt=None, overwrite=False):
            """Save this struct to disk. See strux.save for details."""
            return save(path, self, fmt=fmt, overwrite=overwrite)
        Dataclass.save = _save_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'save', so the "
            f"convenience method .save() will not be available; use "
            f"strux.save(path, obj) instead",
        )
    if "restore" not in fields:
        def _restore_method(self, path, *, fmt=None):
            """Load from disk using this struct as the template. See strux.load."""
            return load(path, template=self, fmt=fmt)
        Dataclass.restore = _restore_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'restore', so the "
            f"convenience method .restore() will not be available; use "
            f"strux.load(path, template=obj) instead",
        )

    # construction checking: the constructor validates the new instance
    # against the schema derived from the field annotations (dtype kind and
    # element dims enforced, leading batch dims free but consistent across
    # fields; see strux.schema). The runtime __init__ deliberately carries
    # no annotations of its own: external runtime checkers that wrap
    # dataclass constructors (such as jaxtyping's import hook) would
    # otherwise enforce the rank-exact field annotations and reject
    # legitimately batched constructions (JAX rebuilds structs through the
    # constructor when unflattening, so vmap/scan/tree-stack results all
    # construct with batch-dim'd leaves). Static type checkers are
    # unaffected: they derive __init__ from the field annotations (PEP 681).
    original_init = Dataclass.__init__
    if check:
        def checked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _validate_struct(self)
        # copy identity but not annotations, and expose an
        # annotation-stripped signature: external checkers must find
        # nothing to enforce, whether they read __annotations__ or
        # inspect.signature (which is also why there is no __wrapped__
        # link back to the annotated original)
        checked_init.__name__ = original_init.__name__
        checked_init.__qualname__ = original_init.__qualname__
        checked_init.__module__ = original_init.__module__
        checked_init.__doc__ = original_init.__doc__
        checked_init.__signature__ = _strip_annotations_signature(
            original_init
        )
        Dataclass.__init__ = checked_init
    else:
        original_init.__annotations__ = {}
        original_init.__signature__ = _strip_annotations_signature(
            original_init
        )

    # done!
    return Dataclass


def _checked_replace(self, **changes):
    """
    Return a copy of this struct with the given fields replaced.

    Replacements whose leaf layout (pytree structure, array shapes and
    dtypes, python scalar types) exactly matches the field they replace
    cannot change the instance's validity, so they are constructed
    directly without revalidation — the common case of swapping
    same-shaped values (e.g. updated parameters each training step) costs
    no solving. Any other replacement constructs through the class (and is
    validated like any direct construction).
    """
    all_fields = dataclasses.fields(self)
    names = {field.name for field in all_fields}
    if not changes.keys() <= names or any(not f.init for f in all_fields):
        # unknown names (or unusual field configs): dataclasses.replace
        # raises the standard errors
        return dataclasses.replace(self, **changes)
    data_fields = set(type(self)._data_fields)
    for name, new in changes.items():
        if name not in data_fields:
            continue    # statics are not validated: never block fast path
        old = getattr(self, name)
        if new is old:
            continue
        if not _same_leaf_layout(old, new):
            return dataclasses.replace(self, **changes)
    obj = object.__new__(type(self))
    for field in all_fields:
        object.__setattr__(
            obj, field.name, changes.get(field.name, getattr(self, field.name)),
        )
    solution = getattr(self, _CACHED_SOLUTION, _UNSOLVED)
    if solution is not _UNSOLVED:
        # identical layouts solve to identical candidates
        object.__setattr__(obj, _CACHED_SOLUTION, solution)
    return obj


def _same_leaf_layout(old, new):
    old_leaves, old_treedef = jax.tree.flatten(old)
    new_leaves, new_treedef = jax.tree.flatten(new)
    if old_treedef != new_treedef:
        return False
    for old_leaf, new_leaf in zip(old_leaves, new_leaves):
        if _is_arraylike_value(old_leaf) and _is_arraylike_value(new_leaf):
            if (
                tuple(old_leaf.shape) != tuple(new_leaf.shape)
                or old_leaf.dtype != new_leaf.dtype
            ):
                return False
        elif type(old_leaf) is not type(new_leaf):
            return False
    return True


def _strip_annotations_signature(fn):
    signature = inspect.signature(fn)
    return signature.replace(
        parameters=[
            param.replace(annotation=inspect.Parameter.empty)
            for param in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )
