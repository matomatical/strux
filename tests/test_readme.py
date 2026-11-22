"""
Import the README's Python blocks in order, in one shared namespace, and
compare their stdout with adjacent console blocks or blocks introduced by
``Output:`` lines.
Run the examples in a subprocess with a time limit so an integration hang
fails with block locations and a traceback instead of stalling pytest.

An output block preceded by an ``<!--output:illustrative-->`` comment is
not compared (used for outputs that cannot be deterministic, such as ones
containing temporary paths). ``<!--code:illustrative-->`` excludes a code
block describing setup in a separate file. Executed blocks are assembled
into a source module so the public import hook can instrument them.
"""

import contextlib
import faulthandler
import importlib
import io
import pathlib
import re
import subprocess
import sys
import tempfile
import traceback
import types

import pytest

import strux

README = pathlib.Path(__file__).parent.parent / "README.md"

_FENCE = re.compile(r"(\s*)(`{3,})(\w*)\s*$")
_ILLUSTRATIVE = "<!--output:illustrative-->"
_ILLUSTRATIVE_CODE = "<!--code:illustrative-->"


def _parse(lines):
    """
    The README as a sequence of events: ("code", lineno, source) for python
    blocks, and ("output", lineno, text) for a non-python block that an
    ``Output:`` line links to the python block before it. Fences may be
    indented (their indent is stripped from the block content). Line numbers
    identify the first line inside each fence, using one-based indexing.
    """
    events = []
    prose = []
    pending_code = False
    i = 0
    while i < len(lines):
        fence = _FENCE.match(lines[i])
        if fence is None:
            prose.append(lines[i])
            i += 1
            continue
        indent, ticks, syntax = fence.groups()
        content = []
        i += 1
        start_line = i + 1
        while not lines[i].lstrip().startswith(ticks):
            content.append(lines[i].removeprefix(indent))
            i += 1
        i += 1
        text = "\n".join(content)
        if syntax == "python":
            illustrative = any(_ILLUSTRATIVE_CODE in line for line in prose)
            if not illustrative:
                events.append(("code", start_line, text))
            pending_code = not illustrative
        else:
            labelled = any("Output:" in line for line in prose)
            adjacent = syntax == "console" and not "".join(prose).strip()
            illustrative = any(_ILLUSTRATIVE in line for line in prose)
            if pending_code and (labelled or adjacent) and not illustrative:
                events.append(("output", start_line, text))
            pending_code = False
        prose = []
    return events


def _run_examples(readme, *, checking=True):
    lines = readme.read_text().splitlines()
    events = _parse(lines)
    expected = {}
    code_line = None
    for kind, lineno, text in events:
        if kind == "code":
            code_line = lineno
        else:
            expected[code_line] = lineno, text

    mismatches = []

    @contextlib.contextmanager
    def block(lineno):
        print(f"Executing {readme}:{lineno}", file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            yield
        if lineno in expected:
            output_line, text = expected[lineno]
            actual = buffer.getvalue()
            if actual.rstrip("\n") != text.rstrip("\n"):
                message = (
                    f"{readme}:{output_line} output mismatch (code at "
                    f"line {lineno}):\nshown:\n{text}\nactual:\n{actual}"
                )
                mismatches.append(message)
                print(message, file=sys.stderr, flush=True)

    support_name = "_strux_readme_harness"
    support = types.ModuleType(support_name)
    support.block = block
    source = [""] * len(lines)
    for kind, lineno, text in events:
        if kind == "code":
            # A with block preserves module scope and the README's line numbers.
            source[lineno - 2] = f"with __import__({support_name!r}).block({lineno}):"
            if not text.strip():
                source[lineno - 2] += " pass"
            for offset, line in enumerate(text.splitlines()):
                source[lineno - 1 + offset] = "    " + line

    with tempfile.TemporaryDirectory(prefix="strux-readme-") as directory:
        module_path = pathlib.Path(directory, "__main__.py")
        module_path.write_text("\n".join(source) + "\n")
        previous_main = sys.modules.pop("__main__", None)
        sys.modules[support_name] = support
        sys.path.insert(0, directory)
        hook = (
            strux.enable_checking("__main__") if checking else contextlib.nullcontext()
        )
        try:
            with hook:
                importlib.import_module("__main__")
        except BaseException:
            diagnostic = traceback.format_exc().replace(str(module_path), str(readme))
            print(diagnostic, file=sys.stderr)
            raise SystemExit(1) from None
        finally:
            sys.path.remove(directory)
            sys.modules.pop(support_name, None)
            sys.modules.pop("__main__", None)
            if previous_main is not None:
                sys.modules["__main__"] = previous_main
    assert not mismatches, (
        f"{len(mismatches)} README output(s) do not match what the code "
        "prints:\n\n" + "\n---\n".join(mismatches)
    )


def _check_readme(readme, *, timeout=60, checking=True):
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                str(readme),
                str(timeout),
                str(int(checking)),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        diagnostics = (error.stderr or b"").decode(errors="replace")
        raise AssertionError(
            f"README examples timed out after {timeout}s.\n{diagnostics}"
        ) from None
    if result.returncode != 0:
        raise AssertionError(
            f"README examples failed (exit {result.returncode}).\n{result.stderr}"
        )


@pytest.mark.parametrize("checking", [False, True], ids=["explicit", "import-hook"])
def test_readme_outputs(checking):
    _check_readme(README, checking=checking)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(float(sys.argv[2]) * 0.8)
    try:
        _run_examples(pathlib.Path(sys.argv[1]), checking=bool(int(sys.argv[3])))
    finally:
        faulthandler.cancel_dump_traceback_later()
