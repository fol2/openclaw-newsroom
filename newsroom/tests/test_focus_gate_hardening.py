from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.sdlc.focus_gate_v2 as focus_gate
import scripts.sdlc.focus_selector as selector


class _Graph:
    def __init__(self, dependents: dict[str, tuple[str, ...]] | None = None) -> None:
        self._dependents = dependents or {}

    def dependent_paths(self, path: str) -> tuple[str, ...]:
        return self._dependents.get(path, ())


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_short_constant_reexport_selects_exact_consumer_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/example/models.py", "X = 7\n")
    _write(
        tmp_path,
        "newsroom/example/__init__.py",
        "from .models import X\n\nOther = 9\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_constant.py",
        "from newsroom.example import X\n\n"
        "def test_constant() -> None:\n    assert X == 7\n",
    )
    _write(
        tmp_path,
        "newsroom/tests/test_unrelated_package_import.py",
        "from newsroom.example import Other\n\n"
        "def test_other() -> None:\n    assert Other == 9\n",
    )
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(
            {
                "newsroom/example/models.py": (
                    "newsroom/example/__init__.py",
                )
            }
        ),
    )

    route = selector.select_focus(
        ("newsroom/example/models.py",),
        repo_root=tmp_path,
    )

    assert route["selected_tests"] == ["newsroom/tests/test_constant.py"]
    assert route["full_health_required"] is False


def test_stateful_route_uses_direct_tests_and_two_bounded_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/authority/example.py", "def commit() -> int:\n    return 1\n")
    _write(
        tmp_path,
        "newsroom/tests/test_example_authority_consumer.py",
        "from newsroom.authority.example import commit\n\n"
        "def test_commit() -> None:\n    assert commit() == 1\n",
    )
    _write(tmp_path, "newsroom/tests/test_authority_migration_compatibility.py")
    _write(tmp_path, "newsroom/tests/test_authority_store_conformance.py")
    for index in range(5):
        _write(tmp_path, f"newsroom/tests/test_authority_unrelated_{index}.py")
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(),
    )

    route = selector.select_focus(
        ("newsroom/authority/example.py",),
        repo_root=tmp_path,
    )

    assert set(route["selected_tests"]) == {
        "newsroom/tests/test_example_authority_consumer.py",
        "newsroom/tests/test_authority_migration_compatibility.py",
        "newsroom/tests/test_authority_store_conformance.py",
    }
    assert not any("unrelated" in path for path in route["selected_tests"])
    assert route["full_health_required"] is False
    assert "bounded_stateful_sentinels:F2" in route["reasons"]


def test_stateful_route_without_direct_evidence_escalates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/authority/uncovered.py", "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_authority_migration_compatibility.py")
    _write(tmp_path, "newsroom/tests/test_authority_store_conformance.py")
    for index in range(5):
        _write(tmp_path, f"newsroom/tests/test_authority_unrelated_{index}.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus(
        ("newsroom/authority/uncovered.py",),
        repo_root=tmp_path,
    )

    assert route["full_health_required"] is True
    assert set(route["selected_tests"]) == {
        "newsroom/tests/test_authority_migration_compatibility.py",
        "newsroom/tests/test_authority_store_conformance.py",
    }
    assert not any("unrelated" in path for path in route["selected_tests"])
    assert "stateful_without_direct_evidence:full_health" in route["reasons"]


def test_discovered_actual_service_consumer_promotes_route_to_f3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "newsroom/feature.py", "def value() -> int:\n    return 1\n")
    _write(
        tmp_path,
        "newsroom/tests/test_feature_neo4j_service.py",
        "from newsroom.feature import value\n\n"
        "def test_service() -> None:\n    assert value() == 1\n",
    )
    monkeypatch.setattr(
        selector,
        "build_dependency_graph",
        lambda _root: _Graph(),
    )

    route = selector.select_focus(
        ("newsroom/feature.py",),
        repo_root=tmp_path,
    )

    assert "F3" in route["gates"]
    assert route["selected_tests"] == []
    assert route["selected_service_tests"] == [
        "newsroom/tests/test_feature_neo4j_service.py"
    ]
    assert "actual_service_consumer:F3" in route["reasons"]


def test_shared_dependency_change_truthfully_selects_research_and_full_health() -> None:
    route = selector.select_focus(("pyproject.toml",))

    assert route["research_required"] is True
    assert route["full_health_required"] is True
    assert route["bootstrap_required"] is True
    assert {"F0", "F1", "F2"} <= set(route["gates"])


def test_research_markdown_remains_documentation_only() -> None:
    route = selector.select_focus(("docs/research/notes.md",))

    assert route["research_required"] is False
    assert route["gates"] == ["F0"]
    assert route["bootstrap_required"] is False


def test_machine_research_evidence_matches_workflow_trigger() -> None:
    route = selector.select_focus(("docs/research/result.json",))

    assert route["research_required"] is True
    assert route["gates"] == ["F0"]
    assert route["selected_tests"] == []
    assert route["bootstrap_required"] is False


def test_f0_uses_locked_interpreter_only_when_bootstrap_is_required() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/focus-gates.yml"
    ).read_text(encoding="utf-8")

    assert "BOOTSTRAP_REQUIRED: ${{ steps.route.outputs.bootstrap_required }}" in workflow
    assert 'if [[ "${BOOTSTRAP_REQUIRED}" == "true" ]]' in workflow
    assert "uv run --no-sync python -m scripts.sdlc.focus_gate_v2" in workflow
    assert "python -m scripts.sdlc.focus_gate_v2" in workflow


@pytest.mark.parametrize(
    "changed",
    (
        "newsroom/control_plane/issue_790_disposition.py",
        "newsroom/control_plane/cycle.py",
        "newsroom/control_plane/graphiti.py",
    ),
)
def test_prepared_canary_parity_is_required_for_pre_provider_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    _write(tmp_path, changed, "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_issue_790_prepared_canary.py")
    _write(tmp_path, "newsroom/tests/test_issue_790_retry_forbidden_safety_state.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus((changed,), repo_root=tmp_path)

    assert "newsroom/tests/test_issue_790_prepared_canary.py" in route["selected_tests"]
    assert (
        "newsroom/tests/test_issue_790_retry_forbidden_safety_state.py"
        in route["selected_tests"]
    )
    assert "prepared_canary_parity:F1" in route["reasons"]


@pytest.mark.parametrize(
    "changed",
    (
        "newsroom/control_plane/issue_790_disposition.py",
        "newsroom/control_plane/issue_790_canary.py",
        "newsroom/control_plane/issue_790_prepared_canary.py",
        "newsroom/control_plane/issue_790_rehearsal.py",
        "scripts/issue_790_live_canary_preflight.py",
        "scripts/issue_790_prepared_canary_rehearsal.py",
    ),
)
def test_prepared_canary_route_selects_model_usage_receipt_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    _write(tmp_path, changed, "VALUE = 1\n")
    _write(tmp_path, "newsroom/tests/test_model_usage_receipts.py")
    monkeypatch.setattr(selector, "build_dependency_graph", lambda _root: _Graph())

    route = selector.select_focus((changed,), repo_root=tmp_path)

    assert "F2" in route["gates"]
    assert "newsroom/tests/test_model_usage_receipts.py" in route["selected_tests"]
    assert "prepared_canary_consumers:F2" in route["reasons"]


def test_f0_validates_yaml_and_shell_syntax(tmp_path: Path) -> None:
    valid_yaml = tmp_path / "valid.yml"
    invalid_yaml = tmp_path / "invalid.yml"
    valid_shell = tmp_path / "valid.sh"
    invalid_shell = tmp_path / "invalid.sh"

    valid_yaml.write_text("jobs:\n  test:\n    runs-on: ubuntu-latest\n", encoding="utf-8")
    invalid_yaml.write_text("jobs: [\n", encoding="utf-8")
    valid_shell.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n", encoding="utf-8")
    invalid_shell.write_text("#!/usr/bin/env bash\nif then\n", encoding="utf-8")

    focus_gate._validate_yaml(valid_yaml)
    focus_gate._validate_shell(valid_shell)
    with pytest.raises(focus_gate.FocusGateError, match="invalid YAML"):
        focus_gate._validate_yaml(invalid_yaml)
    with pytest.raises(focus_gate.FocusGateError, match="invalid shell syntax"):
        focus_gate._validate_shell(invalid_shell)


def test_selected_evidence_uses_fixed_parallel_then_serial_commands(
    tmp_path: Path,
) -> None:
    commands = focus_gate.build_selected_test_commands(
        tmp_path,
        selected_tests=("newsroom/tests/test_b.py", "newsroom/tests/test_a.py"),
        selected_service_tests=("newsroom/tests/test_neo4j_service.py",),
        junit=".focus/pytest.xml",
    )

    assert commands == (
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--assert=plain",
            "-p",
            "no:cacheprovider",
            "-n",
            "4",
            "--dist",
            "worksteal",
            "--max-worker-restart=0",
            "newsroom/tests/test_a.py",
            "newsroom/tests/test_b.py",
            f"--junitxml={tmp_path / '.focus/pytest-provider-free.xml'}",
        ),
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--assert=plain",
            "-p",
            "no:cacheprovider",
            "newsroom/tests/test_neo4j_service.py",
            f"--junitxml={tmp_path / '.focus/pytest-service.xml'}",
        ),
    )


def test_focus_workflow_keeps_finite_budget_for_two_phase_execution() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/focus-gates.yml"
    ).read_text(encoding="utf-8")

    assert "timeout-minutes: 45" in workflow
    assert "--junit .focus/pytest.xml" in workflow


def test_two_phase_junit_preserves_requested_report_path(tmp_path: Path) -> None:
    provider_free = tmp_path / "pytest-provider-free.xml"
    service = tmp_path / "pytest-service.xml"
    requested = tmp_path / "pytest.xml"
    provider_free.write_text(
        '<testsuites><testsuite name="provider-free" tests="2" failures="0" '
        'errors="0" skipped="1" time="1.25" /></testsuites>',
        encoding="utf-8",
    )
    service.write_text(
        '<testsuites><testsuite name="service" tests="1" failures="0" '
        'errors="0" skipped="0" time="2.5" /></testsuites>',
        encoding="utf-8",
    )

    focus_gate._merge_junit_reports(requested, (provider_free, service))

    root = ET.parse(requested).getroot()
    assert root.attrib == {
        "tests": "3",
        "failures": "0",
        "errors": "0",
        "skipped": "1",
        "time": "3.750000",
    }
    assert [suite.attrib["name"] for suite in root] == [
        "provider-free",
        "service",
    ]


def test_failed_phase_cannot_publish_stale_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = tmp_path / "pytest.xml"
    stale = tmp_path / "pytest-provider-free.xml"
    requested.write_text("stale requested", encoding="utf-8")
    stale.write_text("stale phase", encoding="utf-8")
    command = ("pytest", f"--junitxml={stale}")

    monkeypatch.setattr(
        focus_gate,
        "verify_route",
        lambda _root, _route: {
            "research_required": False,
            "bootstrap_required": True,
            "full_health_required": False,
            "selected_tests": ["newsroom/tests/test_a.py"],
            "selected_service_tests": [],
        },
    )
    monkeypatch.setattr(
        focus_gate,
        "build_selected_test_commands",
        lambda *_args, **_kwargs: (command,),
    )
    monkeypatch.setattr(
        focus_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=3),
    )

    assert focus_gate.execute_route(tmp_path, "route.json", junit=requested) == 3
    assert not requested.exists()
    assert not stale.exists()


@pytest.mark.parametrize("route", [selector.select_focus, selector.legacy.select_focus])
def test_internal_publication_is_not_a_public_effect_gate(route) -> None:
    paths = (
        "newsroom/increment10/publication.py",
        "newsroom/tests/test_increment10_publication.py",
    )
    internal = route(paths)
    assert internal["owner_authority_required"] is False
    assert "F4" not in internal["gates"]
    for effect in (
        "deploy/hermes.plist",
        "newsroom/control_plane/keychain.py",
        "scripts/production_operational_admission.py",
    ):
        mixed = route((*paths, effect))
        assert mixed["owner_authority_required"] is True
        assert "F4" in mixed["gates"]
