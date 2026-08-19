from __future__ import annotations


def test_top_level_imports():
    import intelligrate
    import intelligrate.extrapolate
    import intelligrate.subset

    assert intelligrate.__all__ == ["extrapolate", "subset"]
