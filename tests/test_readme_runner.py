"""Failure diagnostics and execution semantics of the README runner."""

import pytest

from test_readme import _check_readme


def test_shared_namespace_and_illustrative_output(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```python\nvalue = 3\n```\n"
        "```python\nprint(value + 1)\n```\n"
        "Output:\n```console\n4\n```\n"
        "```python\nprint('variable output')\n```\n"
        "<!--output:illustrative-->\nOutput:\n```console\nillustration\n```\n"
    )
    _check_readme(readme)


def test_adjacent_console_blocks_are_checked_without_output_label(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```console\npip install example\n```\n"
        "```python\nprint('actual')\n```\n\n"
        "```console\nexpected\n```\n"
    )
    with pytest.raises(AssertionError, match="output mismatch"):
        _check_readme(readme)
    readme.write_text(readme.read_text().replace("expected", "actual"))
    _check_readme(readme)


@pytest.mark.parametrize("later_exception", [False, True])
def test_mismatch_reports_failure_and_source_location(tmp_path, later_exception):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```python\nprint('actual')\n```\n"
        "Output:\n```console\nexpected\n```\n"
        + (
            "```python\nraise RuntimeError('later failure')\n```\n"
            if later_exception
            else ""
        )
    )
    with pytest.raises(AssertionError) as failure:
        _check_readme(readme)
    message = str(failure.value)
    assert "output mismatch" in message
    assert "shown:\nexpected\nactual:\nactual" in message
    assert f"{readme}:6 output mismatch (code at line 2)" in message
    if later_exception:
        assert f'File "{readme}", line 9' in message
        assert "RuntimeError: later failure" in message


def test_hang_reports_active_block_and_earlier_mismatch(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```python\nprint('actual')\n```\n"
        "Output:\n```console\nexpected\n```\n"
        "```python\nimport time\ntime.sleep(30)\n```\n"
    )
    with pytest.raises(AssertionError) as failure:
        _check_readme(readme, timeout=2)
    message = str(failure.value)
    assert "timed out after 2s" in message
    assert "output mismatch" in message
    assert f"Executing {readme}:9" in message


def test_illustrative_setup_is_skipped_without_skipping_following_code(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!--code:illustrative-->\n"
        "```python\nimport nonexistent_example_package\n```\n"
        "```console\nillustrative output\n```\n"
        "```python\nprint('executed')\n```\n"
        "```console\nexecuted\n```\n"
    )
    _check_readme(readme)


def test_import_hook_checks_undecorated_readme_definitions(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "```python\n"
        "import strux\n"
        "@strux.struct\n"
        "class Item:\n"
        "    value: int\n"
        "def identity(value: Item) -> Item:\n"
        "    return value\n"
        "identity(Item('wrong'))\n"
        "```\n"
    )
    _check_readme(readme, checking=False)
    with pytest.raises(AssertionError, match="TypeCheckError"):
        _check_readme(readme, checking=True)
