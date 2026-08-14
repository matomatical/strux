# Changelog

Versions and breaking changes are tracked here from 0.1.0 onward. Earlier
development (0.0.x) is untracked; see the git history.

## 0.2.0 (2026-08-14)

The batch view becomes a first-class, safe API: bounds-aware indexing,
length and iteration, no scalar broadcasting, and cached batch solving.

### Added

* Batched structs support `len(s)` and `for x in s:` (iterating the
  leading batch dimension); module-level `strux.tree_len` and
  `strux.tree_iter`.
* The solved batch-shape candidates are cached on instances at
  construction (or on first query for unflattened/unchecked instances),
  so `.shape` and indexing are O(1) and nested construction no longer
  re-validates already-validated children (measured: nested checked
  construction 17.6µs → 2.9µs, `.shape` 17.3µs → 0.2µs on the dev box).
* `.replace` skips revalidation when the replacement's leaf layout
  (structure, shapes, dtypes, python scalar types) exactly matches the
  field it replaces — such a replace cannot change validity. A side
  effect (provisional, pending the functor-annotations design): layout-
  identical replaces on tree-transformed instances now succeed instead of
  failing validation.
* User-defined `__getitem__`/`__len__`/`__iter__` are preserved with a
  warning (previously a user `__getitem__` was silently overwritten).
* `strux.metadata(tree)` — the metadata mapping `save` records, publicly
  available as the companion of `to_dict`/`from_dict`.
* Pretty printing renders declared dataclass fields only, so the cached
  solver state (in the instance `__dict__`) never appears in renders.

### Changed (breaking)

* A python scalar in a data field admits batch shape `()` only:
  constructing a *batched* struct with a python-scalar field value is now
  a ValidationError (previously scalars were treated as batch-agnostic,
  i.e. silently broadcast). Batched structs carry arrays; scalar leaves
  inside generic registered-pytree fields constrain the same way.
* Indexing is strict: integer batch indices are bounds-checked
  (IndexError out of bounds — previously out-of-bounds indices silently
  clamped, which also made accidental iteration loop forever), tuple
  indices must not exceed the batch rank, and indexing, `len`, or
  iterating an *unbatched* struct raises TypeError (previously indexing
  silently reached into element dimensions, producing invalid structs).

## 0.1.0 (2026-08-14)

Serialisation v2: saved files become self-describing.

### Added

* Saved files carry a metadata mapping alongside the arrays (safetensors
  metadata header / reserved npz entry `__strux__`): structure tags
  (subclass and union-arm identity at each path), literal static field
  values (`repr`/`ast.literal_eval` — no code execution on load), and
  recorded dtypes for leaves npz stores as raw bytes.
* Template-free restore uses the metadata: base-annotated fields holding
  subclasses restore as the saved subclass (resolved by name among imported
  classes), recorded union arms restore directly even when arms leave
  identical array layouts, and literal statics no longer need `statics=`.
* ml_dtypes leaves (e.g. `bfloat16`) now survive an npz round-trip
  (previously the load failed with a raw-void dtype).
* Checkpoint inspection: `strux.describe(path)` and
  `python -m strux <checkpoint>` render any npz/safetensors file's arrays
  plus recorded structure, without constructing structs.
* Python-scalar leaves restore with their python type (previously they
  came back as rank-0 arrays with an instance template).

### Changed (breaking)

* Instance-template restore is strict: saved leaves must match the
  template's shapes and dtypes (previously the saved shapes silently
  replaced the template's), and the file's recorded structure tags must
  agree with the template's structure. This applies to old (metadata-free)
  files too.
* Class-template restore prefers recorded literal statics over field
  defaults (`statics=` still overrides both).
* In npz files the entry name `__strux__` is reserved; a tree with a field
  of that name refuses to save as npz.

Files written by earlier strux versions still load (they simply carry no
metadata, so the pre-0.1.0 template-free limits apply to them).
