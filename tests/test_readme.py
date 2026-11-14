"""
Execute the README's Python blocks in order, in one shared namespace, and
compare their stdout with adjacent console blocks or blocks introduced by
``Output:`` lines.
Run the examples in a subprocess with a time limit so an integration hang
fails with block locations and a traceback instead of stalling pytest.

An output block preceded by an ``<!--output:illustrative-->`` comment is
not compared (used for outputs that cannot be deterministic, such as ones
containing temporary paths).
"""

import contextlib
import faulthandler
import io
import pathlib
import re
import subprocess
import sys

# The working API specification lives in the independent draft during the rebuild.
README = pathlib.Path(__file__).parent.parent / "README-draft.md"

_FENCE = re.compile(r"(\s*)(`{3,})(\w*)\s*$")
_ILLUSTRATIVE = "<!--output:illustrative-->"


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
            events.append(("code", start_line, text))
            pending_code = True
        else:
            labelled = any("Output:" in line for line in prose)
            adjacent = syntax == "console" and not "".join(prose).strip()
            illustrative = any(_ILLUSTRATIVE in line for line in prose)
            if pending_code and (labelled or adjacent) and not illustrative:
                events.append(("output", start_line, text))
            pending_code = False
        prose = []
    return events


def _run_examples(readme):
    events = _parse(readme.read_text().splitlines())
    namespace = {"__name__": "__main__"}
    captured = None
    code_line = None
    mismatches = []
    for kind, lineno, text in events:
        if kind == "code":
            print(f"Executing {readme}:{lineno}", file=sys.stderr, flush=True)
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                source = "\n" * (lineno - 1) + text
                exec(compile(source, str(readme), "exec"), namespace)
            captured = buffer.getvalue()
            code_line = lineno
        elif captured is not None:
            if captured.rstrip("\n") != text.rstrip("\n"):
                message = (
                    f"{readme}:{lineno} output mismatch (code at "
                    f"line {code_line}):\nshown:\n{text}\nactual:\n"
                    f"{captured}"
                )
                mismatches.append(message)
                print(message, file=sys.stderr, flush=True)
            captured = None
    assert not mismatches, (
        f"{len(mismatches)} README output(s) do not match what the code "
        "prints:\n\n" + "\n---\n".join(mismatches)
    )


def _check_readme(readme, *, timeout=60):
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                str(readme),
                str(timeout),
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


def test_readme_outputs():
    _check_readme(README)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(float(sys.argv[2]) * 0.8)
    try:
        _run_examples(pathlib.Path(sys.argv[1]))
    finally:
        faulthandler.cancel_dump_traceback_later()
