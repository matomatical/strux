"""
Zero-code checkpoint inspection: describe the arrays, static fields, and
structure recorded in a saved strux file (or any npz/safetensors file of
named arrays) without constructing any structs.

Command line: `python -m strux <checkpoint>`.
"""

import numpy as np

from strux.serial import _infer_format, _meta_key, _read_file


def describe(path, *, fmt=None) -> str:
    """
    A human-readable description of a saved file: a header line (format,
    metadata version, array and element counts) followed by the tree of
    arrays as `name: dtype[shape]` lines, annotated with the recorded
    class names, literal static values, and None-valued fields where the
    file's metadata records them.

    Works on any npz or safetensors file of named arrays; files saved by
    strux additionally show their recorded structure. Nothing is parsed
    beyond literal values and nothing is constructed, so inspection never
    requires (or runs) the code that saved the file.
    """
    if fmt is None:
        fmt = _infer_format(path)
    d, meta = _read_file(path, fmt)
    # display recorded true dtypes rather than their raw-bytes storage
    for key in list(d):
        name = meta.get(_meta_key("dtype", key))
        if name is not None:
            d[key] = d[key].view(np.dtype(name))
    # group the recorded statics and None-valued fields by parent path
    statics = {}
    nones = {}
    for key, value in meta.items():
        kind, _, key_path = key.partition(" ")
        parent, _, name = key_path.rpartition("/")
        if kind == "static":
            statics.setdefault(parent, []).append((name, value))
        elif kind == "arm" and value == "none":
            nones.setdefault(parent, []).append(name)
    lines = [_header(path, fmt, d, meta)]
    tree = _nest(d)
    if "class" in meta:
        lines.append(meta["class"])
        _render(tree, meta, statics, nones, "", "  ", lines)
    else:
        _render(tree, meta, statics, nones, "", "", lines)
    return "\n".join(lines)


def _header(path, fmt, d, meta):
    version = meta.get("strux")
    tag = f"strux format {version}" if version else "no strux metadata"
    elements = sum(int(np.prod(arr.shape)) for arr in d.values())
    nbytes = sum(arr.nbytes for arr in d.values())
    return (
        f"{path} ({fmt}, {tag}): "
        f"{len(d)} arrays, {elements:,} elements, {_human_bytes(nbytes)}"
    )


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _nest(d):
    """Group the flat '/'-pathed arrays into a nested dict, in file order."""
    root = {}
    for key, arr in d.items():
        node = root
        *parents, last = key.split("/")
        for segment in parents:
            node = node.setdefault(segment, {})
            if not isinstance(node, dict):
                # a foreign file can use one name as both a leaf and a
                # group prefix; show such keys flat rather than crash
                node = root
                last = key
                break
        node[last] = arr
    return root


def _render(node, meta, statics, nones, path, indent, lines):
    for segment, child in node.items():
        child_path = f"{path}/{segment}" if path else segment
        if isinstance(child, dict):
            class_tag = meta.get(_meta_key("class", child_path))
            suffix = f": {class_tag}" if class_tag else ""
            lines.append(f"{indent}{segment}{suffix}")
            _render(
                child, meta, statics, nones, child_path, indent + "  ", lines,
            )
        else:
            lines.append(f"{indent}{segment}: {_leaf_str(child)}")
    for name, value in statics.get(path, []):
        lines.append(f"{indent}{name}: {value} (static)")
    for name in nones.get(path, []):
        if name not in node:
            lines.append(f"{indent}{name}: None")


def _leaf_str(arr):
    shape = " ".join(str(dim) for dim in arr.shape)
    return f"{np.dtype(arr.dtype).name}[{shape}]"
