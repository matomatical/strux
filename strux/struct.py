"""
The @struct decorator: transform an annotated class into a frozen dataclass
registered as a JAX pytree, with unchecked construction and the
convenience members grafted on.
"""

import dataclasses
import functools
import inspect
import typing
import warnings

import jax

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

    Construction and replacement assign values without validating their
    annotations. Use .check(annotation) or checked function boundaries to
    enforce a description.
    """
    if Class is None:
        return functools.partial(
            struct,
            static_fieldnames=static_fieldnames,
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

    # Reconstruct fields directly: __post_init__ and custom constructor
    # behaviour belong to user construction, not JAX tree reconstruction.
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

    from strux.annotate import check as check_value
    from strux.shapes import as_batch

    for name, method in (("check", check_value), ("as_batch", as_batch)):
        if name not in fields and name not in Class.__dict__:
            setattr(Dataclass, name, method)
        else:
            warnings.warn(
                f"{Class.__name__} defines {name!r}; use strux.{name}(obj, ...) instead"
            )

    # These polymorphic operations assign fields; a function checker must
    # not reinterpret their class annotations as element-only contracts.
    Dataclass.__init__.__annotations__ = {}
    Dataclass.__init__.__signature__ = _strip_annotations_signature(Dataclass.__init__)
    return Dataclass


def _strip_annotations_signature(fn):
    signature = inspect.signature(fn)
    return signature.replace(
        parameters=[
            param.replace(annotation=inspect.Parameter.empty)
            for param in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )
