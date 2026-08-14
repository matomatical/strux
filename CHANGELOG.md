# Changelog

Versions and breaking changes are tracked here from 0.1.0 onward. Earlier
development (0.0.x) is untracked; see the git history.

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
