from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_module_cli_help_commands():
    modules = [
        "intelligrate.extrapolate.train",
        "intelligrate.extrapolate.full_fit",
        "intelligrate.extrapolate.full_predict",
        "intelligrate.extrapolate.fixed_param_sweep",
        "intelligrate.subset.cli",
    ]

    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_installed_console_script_help_commands():
    script_dir = Path(sys.executable).parent
    commands = [
        "intelligrate",
    ]

    for command in commands:
        script = script_dir / command
        assert script.exists(), f"Missing installed console script: {script}"
        result = subprocess.run(
            [str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_hierarchical_console_script_help_commands():
    script = Path(sys.executable).parent / "intelligrate"
    commands = [
        [str(script), "--help"],
        [str(script), "subset", "--help"],
        [str(script), "subset", "write-configs", "--help"],
        [str(script), "subset", "distance", "--help"],
        [str(script), "subset", "suggest-k", "--help"],
        [str(script), "subset", "kmedoids", "--help"],
        [str(script), "subset", "ga", "--help"],
        [str(script), "subset", "run-config", "--help"],
        [str(script), "extrapolate", "--help"],
        [str(script), "extrapolate", "write-config", "--help"],
        [str(script), "extrapolate", "train", "--help"],
        [str(script), "extrapolate", "fixed-param-sweep", "--help"],
        [str(script), "extrapolate", "full-fit", "--help"],
        [str(script), "extrapolate", "full-predict", "--help"],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_hierarchical_cli_writes_config_templates(tmp_path):
    script = Path(sys.executable).parent / "intelligrate"

    extrapolate_config = tmp_path / "configs" / "default.yaml"
    result = subprocess.run(
        [str(script), "extrapolate", "write-config", "--out", str(extrapolate_config)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert extrapolate_config.exists()
    config_text = extrapolate_config.read_text()
    assert "intelligrate extrapolate train" in result.stdout
    assert "x_full:" in config_text
    assert "fixed_param_sweep:" in config_text

    subset_dir = tmp_path / "subset-configs"
    result = subprocess.run(
        [str(script), "subset", "write-configs", "--out-dir", str(subset_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (subset_dir / "subset_distance.yaml").exists()
    assert (subset_dir / "subset_k.yaml").exists()
    assert (subset_dir / "subset_kmedoids.yaml").exists()
    assert (subset_dir / "subset_ga.yaml").exists()
    assert (subset_dir / "fixed_include.tsv").exists()
