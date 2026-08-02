"""
Tests for the @struct decorator: field name collision guards and
__str__/__format__ method resolution.
"""

import warnings

import pytest

import strux


# # #
# Field name collision guards


class TestFieldCollisions:
    def test_size_field_warns(self):
        with pytest.warns(UserWarning, match="field named 'size'"):
            @strux.struct
            class HasSize:
                size: int
                x: float

    def test_size_field_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            @strux.struct
            class HasSize:
                size: int
                x: float
        obj = HasSize(size=3, x=1.0)
        assert obj.size == 3

    def test_replace_field_warns(self):
        with pytest.warns(UserWarning, match="field named 'replace'"):
            @strux.struct
            class HasReplace:
                replace: int
                x: float

    def test_replace_field_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            @strux.struct
            class HasReplace:
                replace: int
                x: float
        obj = HasReplace(replace=42, x=1.0)
        assert obj.replace == 42

    def test_no_warning_without_collision(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            @strux.struct
            class Normal:
                x: float
                y: float


# # #
# Format/str/repr method resolution


class TestMethodResolution:
    """
    Test that strux's auto-assigned __format__ delegates to str(self) for
    empty format specs, and uses tree_format parsing for non-empty specs.
    User overrides always win.
    """

    # -- __str__ / __format__ interaction (the interesting part) --

    def test_neither_overridden(self):
        @strux.struct
        class S:
            x: int
        obj = S(x=1)
        expected = strux.to_str(obj)
        assert str(obj) == expected
        assert f'{obj}' == expected
        assert f'{obj:0}' == strux.to_str(obj, max_depth=0)

    def test_str_overridden(self):
        @strux.struct
        class S:
            x: int
            def __str__(self):
                return "custom_str"
        obj = S(x=1)
        assert str(obj) == "custom_str"
        assert f'{obj}' == "custom_str"
        # non-empty spec still uses strux tree_format
        assert f'{obj:0}' == strux.to_str(obj, max_depth=0)

    def test_format_overridden(self):
        @strux.struct
        class S:
            x: int
            def __format__(self, spec):
                return f"custom_format:{spec}"
        obj = S(x=1)
        assert str(obj) == strux.to_str(obj)
        assert f'{obj}' == "custom_format:"
        assert f'{obj:2}' == "custom_format:2"

    def test_str_and_format_overridden(self):
        @strux.struct
        class S:
            x: int
            def __str__(self):
                return "custom_str"
            def __format__(self, spec):
                return "custom_format"
        obj = S(x=1)
        assert str(obj) == "custom_str"
        assert f'{obj}' == "custom_format"
