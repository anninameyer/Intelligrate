from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _discover_src_packages() -> set[str]:
    packages = set()
    for init_file in (ROOT / "src").rglob("__init__.py"):
        package_dir = init_file.parent
        relative = package_dir.relative_to(ROOT / "src")
        packages.add(".".join(relative.parts))
    return packages


def test_pyproject_release_metadata_is_complete():
    project = _read_pyproject()["project"]

    assert project["name"] == "intelligrate"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:\.dev\d+)?", project["version"])
    assert project["description"] == (
        "Diversity-aware sample subsetting and kNN-based extrapolation from one data matrix/layer "
        "to another across complete datasets."
    )
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["license"] == "BSD-3-Clause"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Annina Meyer"}]
    assert "Topic :: Scientific/Engineering :: Bio-Informatics" in project["classifiers"]
    assert {"microbiome", "metagenomics", "multi-omics", "sample-selection", "subsetting"} <= set(
        project["keywords"]
    )
    assert project["urls"]["Repository"] == "https://github.com/anninameyer/Intelligrate"


def test_license_file_matches_pyproject_metadata():
    text = (ROOT / "LICENSE").read_text()

    assert text.startswith("BSD 3-Clause License")
    assert "Copyright (c) 2026, Annina Meyer." in text
    assert 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"' in text


def test_console_script_names_include_workflow_namespace():
    scripts = _read_pyproject()["project"]["scripts"]

    assert scripts == {
        "intelligrate": "intelligrate._cli:main",
        "intelligrate-extrapolate-train": "intelligrate._cli:extrapolate_train",
        "intelligrate-extrapolate-full-fit": "intelligrate._cli:extrapolate_full_fit",
        "intelligrate-extrapolate-full-predict": "intelligrate._cli:extrapolate_full_predict",
        "intelligrate-extrapolate-fixed-param-sweep": "intelligrate._cli:extrapolate_fixed_param_sweep",
        "intelligrate-subset": "intelligrate._cli:subset",
    }


def test_optional_dependencies_cover_maps_and_dev_tools():
    optional = _read_pyproject()["project"]["optional-dependencies"]

    assert "pytest" in optional["dev"]
    assert "tomli; python_version < '3.11'" in optional["dev"]
    assert optional["kmedoids"] == ["scikit-learn-extra"]
    assert optional["maps"] == ["geopandas", "contextily"]


def test_package_discovery_exposes_only_intelligrate_namespaces():
    packages = _discover_src_packages()

    assert packages == {"intelligrate", "intelligrate.extrapolate", "intelligrate.subset"}
    assert not any(pkg.startswith("ko_from_amplicon") for pkg in packages)


def test_requirements_file_has_no_stale_editable_or_legacy_dependency():
    text = (ROOT / "requirements.txt").read_text()

    assert "ko_from_amplicon" not in text
    assert "ko_from_amplicon_agent" not in text
    assert "-e git+" not in text


def test_manifest_excludes_examples_results_and_generated_metadata_from_distributions():
    lines = {
        line.strip()
        for line in (ROOT / "MANIFEST.in").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "recursive-exclude data *" in lines
    assert "recursive-exclude docs *" in lines
    assert "recursive-exclude results *" in lines
    assert "recursive-exclude src *.egg-info" in lines


def test_extrapolate_default_config_uses_existing_dataset_relative_paths():
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())

    for value in cfg["data"].values():
        path = ROOT / "data" / value
        assert path.exists(), f"Missing default config input: {path}"
        assert not str(value).startswith("data/"), "extrapolate config paths are relative to data/"


def test_subset_configs_use_existing_repo_relative_input_paths():
    for config_name in ["subset_distance.yaml", "subset_k.yaml", "subset_ga.yaml"]:
        cfg = yaml.safe_load((ROOT / "configs" / config_name).read_text())
        for key in ["feature_table", "metadata_table"]:
            if key in cfg:
                path = ROOT / cfg[key]
                assert path.exists(), f"Missing {config_name} input {key}: {path}"
                assert str(cfg[key]).startswith("data/HF_sourdough/")


def test_readme_and_tutorial_markdown_links_point_to_existing_repo_files():
    docs = [
        ROOT / "README.md",
        ROOT / "docs/TUTORIAL_extrapolate.md",
        ROOT / "docs/TUTORIAL_subset.md",
    ]
    link_re = re.compile(r"\[[^\]]+\]\(([^)#][^)]+)\)")

    for doc in docs:
        for raw_target in link_re.findall(doc.read_text()):
            if "://" in raw_target or raw_target.startswith("mailto:"):
                continue
            target = raw_target.split("#", 1)[0]
            resolved = (doc.parent / target).resolve()
            assert resolved.exists(), f"Broken link in {doc.relative_to(ROOT)}: {raw_target}"


def test_notebook_links_are_dataset_specific_and_existing():
    readme = (ROOT / "README.md").read_text()

    assert "02_extrapolate_train_evaluate_full_fit_predict.ipynb" not in readme
    for match in re.findall(r"\((docs/notebooks/[^)]+\.ipynb)\)", readme):
        assert (ROOT / match).exists(), f"Missing notebook linked from README: {match}"
    assert "using [data/HF_sourdough/](data/HF_sourdough/)" in readme
    assert "using [data/hmp/](data/hmp/)" in readme
    assert "using [data/indian/](data/indian/)" in readme
    assert "using [data/primates/](data/primates/)" in readme


def test_notebooks_are_cleared_and_portable():
    forbidden = ["/Users/", "/private/", "/home/", "Desktop/git_sourdough", "vscode-notebook-cell"]

    for nb_path in (ROOT / "docs/notebooks").glob("*.ipynb"):
        nb_text = nb_path.read_text()
        for needle in forbidden:
            assert needle not in nb_text, f"{needle!r} found in {nb_path.relative_to(ROOT)}"

        nb = json.loads(nb_text)
        for idx, cell in enumerate(nb["cells"], start=1):
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None, f"Execution count left in {nb_path}:{idx}"
                assert cell.get("outputs") == [], f"Output left in {nb_path}:{idx}"
                source = "".join(cell.get("source", []))
                executable_lines = [
                    line.strip()
                    for line in source.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                assert not any(line.startswith("!pip install") for line in executable_lines)
                assert not any(line.startswith("! pip install") for line in executable_lines)


def test_subset_notebooks_document_optional_map_dependencies():
    for nb_path in (ROOT / "docs/notebooks").glob("01_subset*.ipynb"):
        text = nb_path.read_text()

        assert 'pip install \\"intelligrate[maps]\\"' in text
        assert "HAS_MAP_DEPS" in text
        assert "Skipping optional geographic basemap plot" in text
