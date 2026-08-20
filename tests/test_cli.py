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
        "intelligrate-extrapolate-train",
        "intelligrate-extrapolate-full-fit",
        "intelligrate-extrapolate-full-predict",
        "intelligrate-extrapolate-fixed-param-sweep",
        "intelligrate-subset",
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
        [str(script), "subset", "run", "--help"],
        [str(script), "extrapolate", "--help"],
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
