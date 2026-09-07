from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from newsroom.control_plane.paths import CANONICAL_PROVING_STORE
from newsroom.control_plane import cycle as cycle_module
from newsroom.research.issue_898_ram_cpu import (
    UNOBSERVED,
    bounded_observation_keys,
    bounded_useful_units,
    canonical_json,
    case_admission_generation,
    case_resolve_event,
    decide,
    digest_text,
    fixture_rows,
    measure,
    prepare_event_identity,
    r2_spec_from_event,
    raw_http_cutoff,
    refuse_canonical_store,
    scan_observations,
    sqlite_backup_snapshot,
    summarise_case,
    CLOCK,
    write_proving_store,
    write_r2_spec,
    write_unpublished_store,
)


def test_refuse_canonical_store() -> None:
    with pytest.raises(RuntimeError, match="canonical store"):
        refuse_canonical_store(CANONICAL_PROVING_STORE)


def test_unobserved_is_not_zero() -> None:
    summary = summarise_case(
        "missing",
        [{"status": UNOBSERVED, "outcome": {}}],
    )
    assert summary["max_peak_rss_bytes"] is UNOBSERVED
    assert summary["median_cpu_seconds"] is UNOBSERVED
    assert summary["max_peak_rss_bytes"] != 0


def test_summarise_uses_largest_rss_sample() -> None:
    summary = summarise_case(
        "rust",
        [
            {"status": "OK", "outcome": {}},
            {
                "rss_after_bytes": 50,
                "rss_held_bytes": 100,
                "ru_maxrss_bytes": 200,
                "time_l_maxrss_bytes": 500,
                "user_cpu_seconds": 1,
                "system_cpu_seconds": 0,
                "outcome": {"combined_peak_rss_bytes": 400},
            },
        ],
    )
    assert summary["max_peak_rss_bytes"] == 500


def test_measure_primary_disables_tracemalloc() -> None:
    result = measure(lambda: {"ok": True})
    assert result["tracemalloc_enabled"] is False
    assert result["tracemalloc_peak_bytes"] is UNOBSERVED
    assert result["tracemalloc_current_bytes"] is UNOBSERVED


def test_sqlite_backup_is_consistent_and_not_size_reuse(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE proving_observations(body BLOB)")
    connection.execute("INSERT INTO proving_observations VALUES(?)", (b"abc",))
    connection.commit()
    connection.close()
    first = tmp_path / "snap-a.sqlite3"
    second = tmp_path / "snap-b.sqlite3"
    meta_a = sqlite_backup_snapshot(source, first)
    meta_b = sqlite_backup_snapshot(source, second)
    assert meta_a["status"] == "COPIED"
    assert meta_a["reused_existing_copy"] is False
    assert meta_a["method"] == "sqlite3.Connection.backup"
    assert meta_a["copy_digest"] == meta_b["copy_digest"]
    assert meta_a["observation_count"] == 1
    copied = sqlite3.connect(first)
    assert copied.execute("SELECT body FROM proving_observations").fetchone()[0] == b"abc"
    copied.close()


def test_malformed_and_empty_do_not_claim_queue(tmp_path: Path) -> None:
    unpublished = tmp_path / "unpublished.sqlite3"
    write_unpublished_store(unpublished)
    before = unpublished.stat().st_mtime_ns
    for kind in ("malformed", "empty"):
        proving = tmp_path / f"{kind}.sqlite3"
        write_proving_store(proving, fixture_rows(kind))
        event = prepare_event_identity(str(proving), CLOCK)
        result = case_resolve_event(str(proving), event)
        assert result["status"] in {"OK", "ERROR"}
        assert result["outcome"].get("queue_claimed") is not True
    connection = sqlite3.connect(unpublished)
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    connection.close()
    if "unpublished_graphiti_revision_events" in tables:
        connection = sqlite3.connect(unpublished)
        claimed = connection.execute(
            "SELECT COUNT(*) FROM unpublished_graphiti_revision_events "
            "WHERE state!='QUEUED'"
        ).fetchone()[0]
        connection.close()
        assert claimed == 0
    assert unpublished.stat().st_mtime_ns == before


def test_resolve_uses_prepared_event_and_one_runtime_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("solo"))
    event = prepare_event_identity(str(proving), CLOCK)
    assert event["status"] == "OK"
    assert event["expected_selected_count"] >= 1
    calls: list[int] = []
    real = cycle_module._resolve_graphiti_event_units

    def wrapped(**kwargs: object) -> object:
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(cycle_module, "_resolve_graphiti_event_units", wrapped)
    result = case_resolve_event(str(proving), event)
    assert calls == [1]
    assert result["status"] == "OK"
    assert result["outcome"]["queue_claimed"] is False
    assert result["outcome"]["selected_unit_count"] >= 1
    assert result["tracemalloc_peak_bytes"] is UNOBSERVED


def test_resolve_does_not_open_canonical(tmp_path: Path) -> None:
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("solo"))
    event = prepare_event_identity(str(proving), CLOCK)
    result = case_resolve_event(str(proving), event)
    assert result["status"] == "OK"
    assert result["outcome"]["queue_claimed"] is False
    assert result["outcome"]["selected_unit_count"] >= 1


def test_observation_scan_digest_is_stable(tmp_path: Path) -> None:
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("representative"))
    cutoff = raw_http_cutoff(CLOCK)
    first = scan_observations(str(proving), cutoff)
    second = scan_observations(str(proving), cutoff)
    assert first["manifest_digest"] == second["manifest_digest"]
    assert first["row_count"] >= 1
    assert first["schema"] == "newsroom.issue-898.observation-scan.v1"
    assert first["manifest_digest"].startswith("sha256:")


def test_canonical_json_orders_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert digest_text(canonical_json({"a": 1})) == digest_text('{"a":1}')


def test_admission_generation_is_provider_free() -> None:
    result = case_admission_generation()
    assert result["status"] == "OK"
    assert result["outcome"]["generation_id_present"] is True


def test_decide_holds_without_rust() -> None:
    decision = decide(
        {
            "A2_import_cycle": {
                "max_peak_rss_bytes": 80 * 1024 * 1024,
                "median_cpu_seconds": 0.1,
                "max_retained_rss_bytes": 70 * 1024 * 1024,
            },
            "C5_load_graphiti_units": {
                "max_peak_rss_bytes": 400 * 1024 * 1024,
                "median_cpu_seconds": 1.2,
                "max_retained_rss_bytes": 350 * 1024 * 1024,
            },
        }
    )
    assert decision["go_or_no_go"] == "HOLD"
    assert "exact row selection" not in decision["reason"]


def test_decide_go_when_rust_scan_clears_gate() -> None:
    digest_run = {
        "status": "OK",
        "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)},
    }
    decision = decide(
        {
            "R0_rust_process_baseline": {
                "max_peak_rss_bytes": 2 * 1024 * 1024,
                "runs": [{"status": "OK", "outcome": {}}],
            },
            "R1_python_observation_scan": {
                "max_peak_rss_bytes": 400 * 1024 * 1024,
                "median_cpu_seconds": 2.0,
                "runs": [digest_run],
            },
            "R1_rust_observation_scan": {
                "max_peak_rss_bytes": 40 * 1024 * 1024,
                "median_cpu_seconds": 0.4,
                "runs": [digest_run],
            },
            "R1_rust_e2e_parent": {
                "max_peak_rss_bytes": 50 * 1024 * 1024,
                "median_cpu_seconds": 0.5,
                "runs": [digest_run],
            },
            "R2_bounded_candidate": {
                "runs": [{"status": "HOLD", "mode": "r2", "outcome": {}}],
            },
        }
    )
    assert decision["go_or_no_go"] == "FEASIBILITY_GO"
    assert decision["first_migration_atom"] == "HOLD"
    assert decision["unit_parity_claimed"] is False
    assert decision["rust_total_cpu_seconds"] == 0.9


def test_decide_holds_when_rust_child_cpu_regresses() -> None:
    digest_run = {
        "status": "OK",
        "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)},
    }
    decision = decide(
        {
            "R0_rust_process_baseline": {
                "max_peak_rss_bytes": 2 * 1024 * 1024,
                "runs": [{"status": "OK", "outcome": {}}],
            },
            "R1_python_observation_scan": {
                "max_peak_rss_bytes": 400 * 1024 * 1024,
                "median_cpu_seconds": 1.0,
                "runs": [digest_run],
            },
            "R1_rust_observation_scan": {
                "max_peak_rss_bytes": 40 * 1024 * 1024,
                "median_cpu_seconds": 2.0,
                "runs": [digest_run],
            },
            "R1_rust_e2e_parent": {
                "max_peak_rss_bytes": 50 * 1024 * 1024,
                "median_cpu_seconds": 0.01,
                "runs": [digest_run],
            },
        }
    )
    assert decision["go_or_no_go"] == "HOLD"
    assert decision["rust_total_cpu_seconds"] == 2.01
    assert "child plus e2e parent" in decision["reason"]


def test_decide_no_go_when_rust_does_not_clear_gate() -> None:
    digest_run = {
        "status": "OK",
        "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)},
    }
    decision = decide(
        {
            "R0_rust_process_baseline": {
                "max_peak_rss_bytes": 2 * 1024 * 1024,
                "runs": [{"status": "OK", "outcome": {}}],
            },
            "R1_python_observation_scan": {
                "max_peak_rss_bytes": 100 * 1024 * 1024,
                "median_cpu_seconds": 1.0,
                "runs": [digest_run],
            },
            "R1_rust_observation_scan": {
                "max_peak_rss_bytes": 95 * 1024 * 1024,
                "median_cpu_seconds": 0.9,
                "runs": [digest_run],
            },
            "R1_rust_e2e_parent": {
                "max_peak_rss_bytes": 98 * 1024 * 1024,
                "median_cpu_seconds": 1.0,
                "runs": [digest_run],
            },
        }
    )
    assert decision["go_or_no_go"] == "NO_GO"


def test_prepare_event_retains_unit_refs(tmp_path: Path) -> None:
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("solo"))
    event = prepare_event_identity(str(proving), CLOCK)
    assert event["unit_refs"]
    keys = bounded_observation_keys(event)
    assert keys == [
        (
            event["unit_refs"][0]["proving_run_id"],
            event["source_id"],
            event["unit_refs"][0]["observation_digest"],
        )
    ]
    spec = r2_spec_from_event(event)
    assert "unit_refs" not in spec
    assert "ingest_id" not in spec
    assert "chunk_digest" not in spec
    assert spec["keys"][0]["observation_digest"] == event["unit_refs"][0][
        "observation_digest"
    ]


def test_python_r2_matches_unit_refs_oracle(tmp_path: Path) -> None:
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("large"))
    event = prepare_event_identity(str(proving), CLOCK)
    result = bounded_useful_units(str(proving), event)
    assert result["status"] == "OK"
    assert result["oracle"]["match"] is True
    assert result["unit_count"] == len(event["unit_refs"]) >= 1
    assert result["row_count"] == 1
    assert str(result["manifest_digest"]).startswith("sha256:")


def test_rust_r2_matches_python_digest(tmp_path: Path) -> None:
    import json
    import shutil
    import subprocess

    if shutil.which("cargo") is None:
        pytest.skip("cargo missing")
    proving = tmp_path / "proving.sqlite3"
    write_proving_store(proving, fixture_rows("rss"))
    event = prepare_event_identity(str(proving), CLOCK)
    spec_path = tmp_path / "r2-spec.json"
    write_r2_spec(spec_path, event)
    python = bounded_useful_units(str(proving), event)
    from newsroom.research.issue_898_ram_cpu import RUST_CRATE, RUST_TARGET

    built = RUST_TARGET / "release" / "issue-898-ram-cpu"
    if not built.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(RUST_CRATE / "Cargo.toml"),
            ],
            check=True,
            env={**__import__("os").environ, "CARGO_TARGET_DIR": str(RUST_TARGET)},
        )
    completed = subprocess.run(
        [str(built), "r2", "--db", str(proving), "--spec", str(spec_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(completed.stdout.strip().splitlines()[-1])
    assert rust["status"] == "OK"
    assert rust["outcome"]["manifest_digest"] == python["manifest_digest"]


def test_decide_r2_holds_when_rust_child_cpu_regresses() -> None:
    digest = "sha256:" + ("cd" * 32)
    r2_run = {
        "status": "OK",
        "outcome": {
            "manifest_digest": digest,
            "oracle": {"match": True},
            "row_count": 1,
            "status": "OK",
            "unit_count": 1,
        },
    }
    decision = decide(
        {
            "R0_rust_process_baseline": {
                "max_peak_rss_bytes": 2 * 1024 * 1024,
                "runs": [{"status": "OK", "outcome": {}}],
            },
            "R1_python_observation_scan": {
                "max_peak_rss_bytes": 400 * 1024 * 1024,
                "median_cpu_seconds": 2.0,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R1_rust_observation_scan": {
                "max_peak_rss_bytes": 40 * 1024 * 1024,
                "median_cpu_seconds": 0.4,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R1_rust_e2e_parent": {
                "max_peak_rss_bytes": 50 * 1024 * 1024,
                "median_cpu_seconds": 0.5,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R2_python_bounded_units": {
                "max_peak_rss_bytes": 80 * 1024 * 1024,
                "median_cpu_seconds": 1.0,
                "runs": [r2_run],
            },
            "R2_bounded_candidate": {
                "max_peak_rss_bytes": 10 * 1024 * 1024,
                "median_cpu_seconds": 2.0,
                "runs": [r2_run],
            },
            "R2_rust_e2e_parent": {
                "max_peak_rss_bytes": 12 * 1024 * 1024,
                "median_cpu_seconds": 0.01,
                "runs": [r2_run],
            },
        }
    )
    assert decision["go_or_no_go"] == "FEASIBILITY_GO"
    assert decision["first_migration_atom"] == "HOLD"
    assert decision["r2_hold"] is True
    assert "CPU" in decision["r2_reason"]


def test_decide_r2_go_when_bounded_output_clears_gate() -> None:
    digest = "sha256:" + ("cd" * 32)
    r2_run = {
        "status": "OK",
        "outcome": {
            "manifest_digest": digest,
            "oracle": {"match": True},
            "row_count": 1,
            "status": "OK",
            "unit_count": 1,
        },
    }
    decision = decide(
        {
            "R0_rust_process_baseline": {
                "max_peak_rss_bytes": 2 * 1024 * 1024,
                "runs": [{"status": "OK", "outcome": {}}],
            },
            "R1_python_observation_scan": {
                "max_peak_rss_bytes": 400 * 1024 * 1024,
                "median_cpu_seconds": 2.0,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R1_rust_observation_scan": {
                "max_peak_rss_bytes": 40 * 1024 * 1024,
                "median_cpu_seconds": 0.4,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R1_rust_e2e_parent": {
                "max_peak_rss_bytes": 50 * 1024 * 1024,
                "median_cpu_seconds": 0.5,
                "runs": [{"status": "OK", "outcome": {"manifest_digest": "sha256:" + ("ab" * 32)}}],
            },
            "R2_python_bounded_units": {
                "max_peak_rss_bytes": 80 * 1024 * 1024,
                "median_cpu_seconds": 1.0,
                "runs": [r2_run],
            },
            "R2_bounded_candidate": {
                "max_peak_rss_bytes": 10 * 1024 * 1024,
                "median_cpu_seconds": 0.4,
                "runs": [r2_run],
            },
            "R2_rust_e2e_parent": {
                "max_peak_rss_bytes": 12 * 1024 * 1024,
                "median_cpu_seconds": 0.1,
                "runs": [r2_run],
            },
        }
    )
    assert decision["go_or_no_go"] == "FEASIBILITY_GO"
    assert decision["first_migration_atom"] == "GO"
    assert decision["r2_hold"] is False
    assert decision["unit_parity_claimed"] is True
