"""
The README's example outputs. pytest-codeblocks (the --codeblocks run)
executes the README's python blocks as one merged chain and so checks that
the examples *run*; it cannot compare each example's printed output against
the ``Output:`` block shown beneath it (one merged chain has one stdout).
This test re-executes the python blocks in order, in one shared namespace
(the same semantics as the chain), and compares what each block prints with
the console block that follows it under an ``Output:`` line.

An output block preceded by an ``<!--output:illustrative-->`` comment is
not compared (used for outputs that cannot be deterministic, such as ones
containing temporary paths).
"""

import contextlib
import io
import pathlib
import re

README = pathlib.Path(__file__).parent.parent / "README.md"

_FENCE = re.compile(r"(\s*)(`{3,})(\w*)\s*$")
_ILLUSTRATIVE = "<!--output:illustrative-->"


def _parse(lines):
    """
    The README as a sequence of events: ("code", lineno, source) for python
    blocks, and ("output", lineno, text) for a non-python block that an
    ``Output:`` line links to the python block before it. Fences may be
    indented (their indent is stripped from the block content, matching
    pytest-codeblocks).
    """
    events = []
    prose = []
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
        while not lines[i].lstrip().startswith(ticks):
            content.append(lines[i].removeprefix(indent))
            i += 1
        i += 1
        text = "\n".join(content)
        if syntax == "python":
            events.append(("code", i, text))
        elif any("Output:" in line for line in prose) and not any(
            _ILLUSTRATIVE in line for line in prose
        ):
            events.append(("output", i, text))
        prose = []
    return events


def test_readme_outputs():
    events = _parse(README.read_text().splitlines())
    namespace = {"__name__": "__main__"}
    captured = None
    code_line = None
    mismatches = []
    for kind, lineno, text in events:
        if kind == "code":
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                exec(compile(text, str(README), "exec"), namespace)
            captured = buffer.getvalue()
            code_line = lineno
        elif captured is not None:
            if captured.rstrip("\n") != text.rstrip("\n"):
                mismatches.append(
                    f"README.md block ending line {lineno} (code ending "
                    f"line {code_line}):\nshown:\n{text}\nactual:\n"
                    f"{captured}"
                )
            captured = None
    assert not mismatches, (
        f"{len(mismatches)} README output(s) do not match what the code "
        "prints:\n\n" + "\n---\n".join(mismatches)
    )
