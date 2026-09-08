from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Iterable, Mapping, Sequence

from .classify_change import (
    ChangedPath,
    changed_paths,
    resolve_commit,
    resolve_tree,
    verify_exact_clean_checkout,
)
from .dependencies import DependencyError, build_dependency_graph, module_name_for_path


SCHEMA_VERSION = "newsroom.sdlc.focus-route.v1"
FOCUS_CONTRACT_VERSION = "focus-gates-v1"
_MAX_MANIFEST_BYTES = 1_048_576

_CONTROL_PATTERNS = (
    ".sdlc/**",
    ".github/workflows/**",
    "scripts/sdlc/focus_gate.py",
    "newsroom/tests/test_focus_gate.py",
    "newsroom/tests/test_ci_workflow.py",
    "newsroom/tests/test_sdlc_contract.py",
    "newsroom/tests/test_sdlc_evidence_workflow.py",
)
_CONTROL_TESTS = (
    "newsroom/tests/test_focus_gate.py",
    "newsroom/tests/test_ci_workflow.py",
    "newsroom/tests/test_sdlc_contract.py",
    "newsroom/tests/test_sdlc_evidence_workflow.py",
    "newsroom/tests/test_pr_lifecycle.py",
)
_RESEARCH_PATTERNS = (
    "docs/research/**",
    "scripts/graphiti_combined_temporal_extraction.py",
    "scripts/graphiti_deterministic_work.py",
    "scripts/graphiti_runtime_calibration.py",
    "newsroom/tests/test_graphiti_core_0293_*.py",
    "newsroom/tests/test_graphiti_sdk_no_tool_calibration.py",
    "newsroom/tests/test_graphiti_combined_temporal_*.py",
    "newsroom/tests/test_graphiti_donor_identities.py",
    "newsroom/tests/test_graphiti_deterministic_work.py",
)
_RESEARCH_TEST_GLOBS = tuple(
    pattern for pattern in _RESEARCH_PATTERNS if pattern.startswith("newsroom/tests/")
)
_SERVICE_PATTERNS = (
    "newsroom/projection/neo4j/**",
    "newsroom/*neo4j*.py",
    "newsroom/**/*neo4j*.py",
    "newsroom/tests/test_*_neo4j_service.py",
)
_STATEFUL_PATTERNS = (
    "newsroom/authority/**",
    "newsroom/integrated/**",
    "newsroom/projection/models.py",
    "newsroom/projection/policy.py",
    "newsroom/schemas/**",
    "newsroom/**/*migration*.py",
    "newsroom/**/*_migrations.py",
)
_F4_PATTERNS = (
    "deploy/**",
    "release/**",
    ".github/workflows/deploy*.yml",
    ".github/workflows/release*.yml",
    "newsroom/production_admission/**",
    "scripts/production_operational_admission.py",
    "newsroom/**/*credential*.py",
    "newsroom/**/*keychain*.py",
)
_SHARED_BREADTH_PATTERNS = (
    "pyproject.toml",
    "uv.lock",
    "newsroom/tests/conftest.py",
)
_EXECUTABLE_SUFFIXES = frozenset(
    {".py", ".toml", ".json", ".yml", ".yaml", ".sh", ".bash", ".sql"}
)
_IGNORED_IMPORT_ROOTS = frozenset({"newsroom", "scripts"})


class FocusGateError(ValueError):
    """Raised when a Focus Gate manifest cannot be trusted or executed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _is_documentation(path: str) -> bool:
    return (
        path.endswith(".md")
        or path.startswith("docs/")
        and Path(path).suffix.lower() not in _EXECUTABLE_SUFFIXES
    )


def _is_research(path: str) -> bool:
    return _matches(path, _RESEARCH_PATTERNS)


def _is_test(path: str) -> bool:
    return path.startswith("newsroom/tests/test_") and path.endswith(".py")


def _is_service_test(path: str) -> bool:
    return _is_test(path) and path.endswith("_neo4j_service.py")


def _existing(repo_root: Path | None, paths: Iterable[str]) -> set[str]:
    values = set(paths)
    if repo_root is None:
        return values
    return {
        path
        for path in values
        if path == "newsroom/tests" or (repo_root / path.split("::", 1)[0]).is_file()
    }


def _test_files(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root / "newsroom" / "tests"
    if root.is_symlink() or not root.is_dir():
        raise FocusGateError("test root is missing or symlinked")
    return tuple(
        path
        for path in sorted(root.rglob("test_*.py"))
        if path.is_file() and not path.is_symlink()
    )


def _imported_modules(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            imports.update(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return imports


def _defined_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and len(node.name) >= 5
        and node.name not in {"main", "parse", "build", "create", "validate"}
    }


def _module_family(
    repo_root: Path,
    source_paths: Sequence[str],
) -> tuple[set[str], set[str], set[str], bool]:
    direct: set[str] = set()
    dependents: set[str] = set()
    reexport_packages: set[str] = set()
    unresolved = False
    graph = None
    try:
        graph = build_dependency_graph(repo_root)
    except DependencyError:
        unresolved = True
    for path in source_paths:
        module = module_name_for_path(path)
        if module is None:
            continue
        direct.add(module)
        if graph is None:
            continue
        try:
            for dependent in graph.dependent_paths(path):
                dependent_module = module_name_for_path(dependent)
                if dependent_module is None:
                    continue
                if dependent.endswith("/__init__.py"):
                    reexport_packages.add(dependent_module)
                else:
                    dependents.add(dependent_module)
        except DependencyError:
            unresolved = True
    return direct, dependents, reexport_packages, unresolved


def _imports_module(imported: str, selected: str) -> bool:
    if imported in _IGNORED_IMPORT_ROOTS or selected in _IGNORED_IMPORT_ROOTS:
        return False
    return imported == selected or imported.startswith(selected + ".")


def _discover_tests(repo_root: Path, source_paths: Sequence[str]) -> tuple[set[str], bool]:
    direct, dependents, reexports, unresolved = _module_family(
        repo_root,
        source_paths,
    )
    names: set[str] = set()
    for relative in source_paths:
        path = repo_root / relative
        if path.is_file() and not path.is_symlink():
            names.update(_defined_names(path))

    selected: set[str] = set()
    reexport_smoke: dict[str, str] = {}
    satisfied_reexports: set[str] = set()
    for path in _test_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        if _is_research(relative):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=relative)
        except (OSError, SyntaxError, UnicodeError):
            unresolved = True
            continue
        imports = _imported_modules(tree)
        direct_hit = any(
            _imports_module(imported, module)
            for imported in imports
            for module in direct
        )
        dependent_hit = any(
            _imports_module(imported, module)
            for imported in imports
            for module in dependents
        )
        imported_reexports = {
            module
            for module in reexports
            if any(_imports_module(imported, module) for imported in imports)
        }
        name_hit = bool(imported_reexports) and any(name in text for name in names)
        for module in imported_reexports - satisfied_reexports:
            reexport_smoke.setdefault(module, relative)
        if direct_hit or dependent_hit or name_hit:
            selected.add(relative)
            if name_hit:
                satisfied_reexports.update(imported_reexports)
                for module in imported_reexports:
                    reexport_smoke.pop(module, None)

    # Keep one deterministic package-import smoke only when no changed-symbol
    # consumer already proves the re-export boundary.
    selected.update(reexport_smoke.values())
    return selected, unresolved


def _glob_tests(repo_root: Path | None, patterns: Sequence[str]) -> set[str]:
    if repo_root is None:
        return set(patterns)
    values: set[str] = set()
    for pattern in patterns:
        for path in (repo_root / "newsroom" / "tests").glob(pattern):
            if path.is_file() and not path.is_symlink():
                values.add(path.relative_to(repo_root).as_posix())
    return values


def validate_focus_contract_data(data: object) -> Mapping[str, object]:
    if not isinstance(data, dict):
        raise FocusGateError("SDLC contract must be a table")
    global_config = data.get("global")
    strategy = data.get("test_strategy")
    focus = data.get("focus")
    if not all(isinstance(item, dict) for item in (global_config, strategy, focus)):
        raise FocusGateError("Focus Gate contract tables are missing")
    expected = {
        "schema_version": "newsroom.sdlc.focus-gates.v1",
        "contract_version": FOCUS_CONTRACT_VERSION,
        "status": "accepted",
        "issue": 799,
        "route_schema": ".sdlc/focus-route.schema.json",
        "ordinary_pr_workflow": ".github/workflows/focus-gates.yml",
        "full_health_workflow": ".github/workflows/evidence.yml",
        "research_workflow": ".github/workflows/ci.yml",
        "ordinary_evidence_job_count": 1,
        "documentation_dependency_bootstraps": 0,
        "executable_dependency_bootstraps": 1,
        "default_gates": ["F0"],
        "full_health_events": ["merge_group", "schedule", "workflow_dispatch"],
        "research_events": ["pull_request_paths", "schedule", "workflow_dispatch"],
        "blocking_selector": "deterministic",
        "provider_calls_implicit": False,
        "human_default_required": False,
        "human_exception_gates": ["F4"],
    }
    if focus != expected:
        raise FocusGateError("Focus Gate contract differs from the accepted policy")
    if global_config.get("full_suite_is_default") is not False:
        raise FocusGateError("ordinary full-suite default must be false")
    if strategy.get("full_suite_blocking_default") is not False:
        raise FocusGateError("blocking full-suite default must be false")
    return focus


def load_focus_contract(repo_root: str | Path) -> Mapping[str, object]:
    root = Path(repo_root).resolve()
    source = root / ".sdlc" / "gates.toml"
    if source.is_symlink() or not source.is_file():
        raise FocusGateError("accepted SDLC contract is missing or symlinked")
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    focus = validate_focus_contract_data(data)
    for key in (
        "route_schema",
        "ordinary_pr_workflow",
        "full_health_workflow",
        "research_workflow",
    ):
        selected = root / str(focus[key])
        if selected.is_symlink() or not selected.is_file():
            raise FocusGateError(f"Focus Gate contract file is missing: {focus[key]}")
    return focus


def _manifest_without_digest(
    *,
    changed: Sequence[str],
    base_sha: str,
    head_sha: str,
    base_tree_sha: str,
    head_tree_sha: str,
    gates: Sequence[str],
    tests: Sequence[str],
    service_tests: Sequence[str],
    research_required: bool,
    full_health_required: bool,
    owner_authority_required: bool,
    bootstrap_required: bool,
    reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": FOCUS_CONTRACT_VERSION,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "base_tree_sha": base_tree_sha,
        "head_tree_sha": head_tree_sha,
        "changed_paths": list(changed),
        "gates": list(gates),
        "selected_tests": list(tests),
        "selected_service_tests": list(service_tests),
        "research_required": research_required,
        "full_health_required": full_health_required,
        "owner_authority_required": owner_authority_required,
        "bootstrap_required": bootstrap_required,
        "reasons": list(reasons),
        "execution_budget": {
            "focus_gate_jobs": 1,
            "dependency_bootstraps": 1 if bootstrap_required else 0,
        },
    }


def select_focus(
    paths: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    base_sha: str = "0" * 40,
    head_sha: str = "1" * 40,
    base_tree_sha: str = "2" * 40,
    head_tree_sha: str = "3" * 40,
) -> dict[str, object]:
    root = None if repo_root is None else Path(repo_root).resolve()
    changed = tuple(sorted(set(paths)))
    if not changed:
        raise FocusGateError("a Focus Gate route requires at least one changed path")

    reasons: set[str] = set()
    gates = {"F0"}
    tests: set[str] = set()
    service_tests: set[str] = set()
    executable_paths = [
        path
        for path in changed
        if not _is_documentation(path)
    ]
    source_paths = [
        path
        for path in changed
        if path.endswith(".py")
        and path.startswith(("newsroom/", "scripts/"))
        and not _is_test(path)
    ]

    research_paths = [path for path in executable_paths if _is_research(path)]
    normal_executable = [path for path in executable_paths if not _is_research(path)]
    research_required = bool(research_paths)
    research_only = bool(research_paths) and not normal_executable
    full_health_required = _matches_any(changed, _SHARED_BREADTH_PATTERNS)
    owner_authority_required = _matches_any(changed, _F4_PATTERNS)
    service_required = _matches_any(changed, _SERVICE_PATTERNS)
    stateful_required = _matches_any(changed, _STATEFUL_PATTERNS)
    control_required = _matches_any(changed, _CONTROL_PATTERNS)

    if all(_is_documentation(path) for path in changed):
        reasons.add("documentation_only:F0")
    if research_required:
        reasons.add("research_inputs_changed:isolated_lane")
    if normal_executable:
        gates.add("F1")
        reasons.add("executable_change:F1")
    if stateful_required or control_required:
        gates.add("F2")
        reasons.add(
            "stateful_contract:F2" if stateful_required else "sdlc_control:F2"
        )
    if service_required:
        gates.add("F3")
        reasons.add("actual_neo4j_semantics:F3")
    if owner_authority_required:
        gates.add("F4")
        reasons.add("irreversible_or_public_effect:F4")

    for path in changed:
        if _is_test(path) and not _is_research(path):
            (service_tests if _is_service_test(path) else tests).add(path)

    if control_required:
        tests.update(_existing(root, _CONTROL_TESTS))

    if source_paths and not research_only and root is not None:
        discovered, unresolved = _discover_tests(root, source_paths)
        for path in discovered:
            (service_tests if _is_service_test(path) else tests).add(path)
        if discovered:
            gates.add("F2")
            reasons.add("repository_import_or_symbol_consumers:F2")
        if unresolved:
            full_health_required = True
            reasons.add("unresolved_dependency_analysis:full_health")

    if stateful_required:
        tests.update(
            _glob_tests(
                root,
                (
                    "test_*migration*.py",
                    "test_authority_*.py",
                    "test_*replay*.py",
                ),
            )
        )

    if service_required:
        discovered_service = _glob_tests(root, ("test_*_neo4j_service.py",))
        if not service_tests:
            service_tests.update(discovered_service)
            reasons.add("service_fallback_inventory:F3")

    known_roots = (
        "newsroom/",
        "scripts/",
        "docs/",
        ".github/",
        ".sdlc/",
    )
    unknown_executable = [
        path
        for path in normal_executable
        if not path.startswith(known_roots) and path not in {"pyproject.toml", "uv.lock"}
    ]
    if unknown_executable:
        full_health_required = True
        reasons.add("unknown_executable_path:full_health")

    tests = _existing(root, tests)
    service_tests = _existing(root, service_tests)
    if normal_executable and not research_only and not (
        tests or service_tests or full_health_required
    ):
        full_health_required = True
        reasons.add("no_defensible_focused_test:full_health")

    if full_health_required:
        gates.update({"F1", "F2"})
        reasons.add("cross_cutting_or_unresolved:full_health")

    if research_only:
        tests.clear()
        service_tests.clear()
        full_health_required = False
        service_required = False
        gates = {"F0"}

    bootstrap_required = bool(
        tests or service_tests or full_health_required or service_required
    )
    body = _manifest_without_digest(
        changed=changed,
        base_sha=base_sha,
        head_sha=head_sha,
        base_tree_sha=base_tree_sha,
        head_tree_sha=head_tree_sha,
        gates=sorted(gates),
        tests=sorted(tests),
        service_tests=sorted(service_tests),
        research_required=research_required,
        full_health_required=full_health_required,
        owner_authority_required=owner_authority_required,
        bootstrap_required=bootstrap_required,
        reasons=sorted(reasons),
    )
    return {**body, "manifest_digest": _digest(body)}


def _matches_any(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(_matches(path, patterns) for path in paths)


def validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FocusGateError("manifest must be an object")
    required = {
        "schema_version",
        "contract_version",
        "base_sha",
        "head_sha",
        "base_tree_sha",
        "head_tree_sha",
        "changed_paths",
        "gates",
        "selected_tests",
        "selected_service_tests",
        "research_required",
        "full_health_required",
        "owner_authority_required",
        "bootstrap_required",
        "reasons",
        "execution_budget",
        "manifest_digest",
    }
    if set(value) != required:
        raise FocusGateError("manifest shape differs from the accepted schema")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FocusGateError("unsupported Focus Gate schema")
    if value["contract_version"] != FOCUS_CONTRACT_VERSION:
        raise FocusGateError("unsupported Focus Gate contract")
    for field in ("changed_paths", "gates", "selected_tests", "selected_service_tests", "reasons"):
        items = value[field]
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            or items != sorted(set(items))
        ):
            raise FocusGateError(f"{field} must be a canonical string set")
    if not value["changed_paths"] or "F0" not in value["gates"]:
        raise FocusGateError("manifest must contain changed paths and F0")
    if any(gate not in {"F0", "F1", "F2", "F3", "F4"} for gate in value["gates"]):
        raise FocusGateError("manifest contains an unknown gate")
    for field in (
        "research_required",
        "full_health_required",
        "owner_authority_required",
        "bootstrap_required",
    ):
        if not isinstance(value[field], bool):
            raise FocusGateError(f"{field} must be boolean")
    for field in ("base_sha", "head_sha", "base_tree_sha", "head_tree_sha"):
        identity = value[field]
        if (
            not isinstance(identity, str)
            or len(identity) != 40
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise FocusGateError(f"{field} must be a lowercase Git SHA")
    gates = set(value["gates"])
    if value["selected_service_tests"] and "F3" not in gates:
        raise FocusGateError("service tests require F3")
    if value["owner_authority_required"] and "F4" not in gates:
        raise FocusGateError("owner authority requires F4")
    if value["full_health_required"] and not {"F1", "F2"} <= gates:
        raise FocusGateError("full health requires F1 and F2")
    if value["research_required"] and not value["bootstrap_required"]:
        if value["selected_tests"] or value["selected_service_tests"]:
            raise FocusGateError("isolated research cannot select ordinary tests")
    budget = value["execution_budget"]
    expected_budget = {
        "focus_gate_jobs": 1,
        "dependency_bootstraps": 1 if value["bootstrap_required"] else 0,
    }
    if budget != expected_budget:
        raise FocusGateError("execution budget is not canonical")
    unsigned = dict(value)
    digest = unsigned.pop("manifest_digest")
    if digest != _digest(unsigned):
        raise FocusGateError("manifest digest mismatch")
    return value


def _load_manifest(path: str | Path) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FocusGateError("manifest file is missing or symlinked")
    payload = candidate.read_bytes()
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise FocusGateError("manifest is too large")
    parsed = json.loads(payload.decode("utf-8"))
    validated = validate_manifest(parsed)
    if payload != canonical_json_bytes(validated) + b"\n":
        raise FocusGateError("manifest is not canonical JSON")
    return validated


def _write_manifest(path: str | Path, value: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FocusGateError("manifest output already exists")
    target.write_bytes(canonical_json_bytes(value) + b"\n")


def _changed_path_strings(changes: Sequence[ChangedPath]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for change in changes
                for path in change.classified_paths()
            }
        )
    )


def build_route(
    repo_root: str | Path,
    *,
    base_reference: str,
    head_reference: str,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    load_focus_contract(root)
    base_sha = resolve_commit(root, base_reference)
    head_sha = resolve_commit(root, head_reference)
    head_tree_sha = resolve_tree(root, head_sha)
    verify_exact_clean_checkout(root, head_sha=head_sha, head_tree_sha=head_tree_sha)
    return select_focus(
        _changed_path_strings(changed_paths(root, base_sha, head_sha)),
        repo_root=root,
        base_sha=base_sha,
        head_sha=head_sha,
        base_tree_sha=resolve_tree(root, base_sha),
        head_tree_sha=head_tree_sha,
    )


def _git_diff_check(root: Path, base_sha: str, head_sha: str) -> None:
    completed = subprocess.run(
        ("git", "diff", "--check", base_sha, head_sha, "--"),
        cwd=root,
        check=False,
    )
    if completed.returncode:
        raise FocusGateError("git diff integrity failed")


def verify_route(repo_root: str | Path, route_path: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    load_focus_contract(root)
    route = _load_manifest(route_path)
    current_head = resolve_commit(root, "HEAD")
    current_tree = resolve_tree(root, current_head)
    if (route["head_sha"], route["head_tree_sha"]) != (current_head, current_tree):
        raise FocusGateError("manifest does not describe the checked-out head")
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=no"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode or status.stdout:
        raise FocusGateError("tracked checkout differs from the routed head")
    _git_diff_check(root, str(route["base_sha"]), current_head)

    for relative in route["changed_paths"]:
        path = root / str(relative)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise FocusGateError(f"unsupported changed entry: {relative}")
        suffix = path.suffix.lower()
        if suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8"))

    for selected in (*route["selected_tests"], *route["selected_service_tests"]):
        if selected == "newsroom/tests":
            continue
        if not (root / str(selected).split("::", 1)[0]).is_file():
            raise FocusGateError(f"selected test is missing: {selected}")
    return route


def _junit_path(root: Path, junit: str | Path) -> Path:
    report = Path(junit)
    if not report.is_absolute():
        report = root / report
    report.parent.mkdir(parents=True, exist_ok=True)
    return report


def execute_full_health(
    repo_root: str | Path,
    *,
    junit: str | Path,
) -> int:
    root = Path(repo_root).resolve()
    load_focus_contract(root)
    report = _junit_path(root, junit)
    command = (
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
        "newsroom/tests",
        *(f"--ignore-glob={pattern}" for pattern in _RESEARCH_TEST_GLOBS),
        f"--junitxml={report}",
    )
    return subprocess.run(command, cwd=root, env=os.environ.copy(), check=False).returncode


def execute_route(
    repo_root: str | Path,
    route_path: str | Path,
    *,
    junit: str | Path,
) -> int:
    root = Path(repo_root).resolve()
    route = verify_route(root, route_path)
    if route["research_required"] and not route["bootstrap_required"]:
        return 0
    if route["full_health_required"]:
        return execute_full_health(root, junit=junit)
    selectors = tuple(
        sorted(set(route["selected_tests"]) | set(route["selected_service_tests"]))
    )
    if not selectors:
        return 0
    report = _junit_path(root, junit)
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--assert=plain",
        "-p",
        "no:cacheprovider",
        *selectors,
        f"--junitxml={report}",
    )
    return subprocess.run(command, cwd=root, env=os.environ.copy(), check=False).returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route and execute Newsroom Focus Gates")
    parser.add_argument("--repo-root", default=".")
    commands = parser.add_subparsers(dest="command", required=True)

    route = commands.add_parser("route")
    route.add_argument("--base", required=True)
    route.add_argument("--head", required=True)
    route.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--route", required=True)

    execute = commands.add_parser("execute")
    execute.add_argument("--route", required=True)
    execute.add_argument("--junit", required=True)

    full_health = commands.add_parser("full-health")
    full_health.add_argument("--junit", required=True)

    summary = commands.add_parser("summary")
    summary.add_argument("--route", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "route":
            value = build_route(
                arguments.repo_root,
                base_reference=arguments.base,
                head_reference=arguments.head,
            )
            _write_manifest(arguments.output, value)
        elif arguments.command == "verify":
            value = verify_route(arguments.repo_root, arguments.route)
        elif arguments.command == "execute":
            return execute_route(
                arguments.repo_root,
                arguments.route,
                junit=arguments.junit,
            )
        elif arguments.command == "full-health":
            return execute_full_health(
                arguments.repo_root,
                junit=arguments.junit,
            )
        else:
            value = _load_manifest(arguments.route)
        sys.stdout.write(canonical_json_bytes(value).decode("utf-8") + "\n")
        return 0
    except (
        FocusGateError,
        json.JSONDecodeError,
        OSError,
        UnicodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(f"FOCUS_GATE_ERROR:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
