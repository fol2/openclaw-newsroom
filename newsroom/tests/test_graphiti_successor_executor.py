from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from newsroom.control_plane.graphiti_steady_state import (
    validate_graphiti_campaign_packet,
)
from newsroom.tests.test_graphiti_steady_state import (
    _authority_store,
    _bounded_candidate_packet,
    _nonterminal_obligation,
    _seed_proving_accountability,
    _stores,
)
from scripts import issue_895_graphiti_executor as executor


GRANT_BODY = (
    "## Standing bounded autonomous F4 programme\n"
    "\n"
    "This supersedes only the requirement to return for a new owner comment "
    "before every successor F4 execution. It does not retry or re-authorise "
    "the event consumed by comment `5520346023`.\n"
    "\n"
    "After the authorised provider-free correction has truthfully disposed "
    "the existing `500000` microunit reservation, retained the consumed event "
    "outside every successor cohort, and sealed a new exact-main "
    "`READY_FOR_OWNER_DECISION` packet, the agent may execute and continue "
    "the #895 F4 loop autonomously.\n"
    "\n"
    "Continuation is allowed only while each successor packet is "
    "machine-proved to use the same accepted source/selection policy, "
    "canonical authority and graph destination, qualified "
    "provider/transport/model/embedding route, and existing admission "
    "semantics; contains only fresh never-attempted events; keeps zero retry "
    "and zero fallback; and retains the same or narrower rate, wall-time, "
    "spend and stop bounds. Across the whole programme, including the "
    "consumed attempt, caps do not reset: at most `218` event starts and "
    "`109000000` GBP microunits of reservations/actual spend.\n"
    "\n"
    "After any stop, retain exact terminal evidence, reconcile every claim, "
    "provider usage, spend, admission and graph effect, make only the "
    "smallest provider-free ordinary correction, reseal exact main, and "
    "continue without another owner comment only when all prior effects are "
    "fully and unambiguously classified. Never requeue or retry an attempted "
    "event.\n"
    "\n"
    "Stop for owner judgement only on unknown/partial effect, irreconcilable "
    "accounting, changed provider/credential or authority policy, destructive "
    "migration, broader source/effect/cap scope, publication/Production "
    "Operational Admission, 11R or #151. This grants no continuous service "
    "load, publication or #151 activation."
)
GRANT_BODY_DIGEST = (
    "ab068ff3c70bc8a09254357026c89ddda64af2a26f4e951513d8bbfde3092459"
)
ISSUE_URL = "https://api.github.com/repos/fol2/newsroom/issues/895"
PROGRESS_BODY = (
    "Phase C report. Coordination grants no live effect. "
    "Implementation-intent review; campaign_authorised remains false."
)


class CommentView:
    def __init__(self, comments: list[dict[str, object]]) -> None:
        self.comments = comments

    def __call__(self) -> list[dict[str, object]]:
        return list(self.comments)


class DispatchStub:
    def __init__(self, *, before_fence=None) -> None:
        self.count = 0
        self.before_fence = before_fence

    def __call__(self, packet, owner_f4_fence, proving, unpublished):
        self.count += 1
        if self.before_fence is not None:
            self.before_fence(packet)
        owner_f4_fence(packet)
        return {"effect_performed": False, "event_rows_written": 0, "spend_rows_written": 0}


@pytest.fixture(autouse=True)
def _never_subprocess_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def guarded(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        if command and command[0] == "gh":
            pytest.fail("tests must not subprocess gh")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "run", guarded)


def _grant(**overrides: object) -> dict[str, object]:
    comment: dict[str, object] = {
        "id": 5522755269,
        "user": {"login": "fol2", "id": 105634418},
        "author_association": "OWNER",
        "issue_url": ISSUE_URL,
        "body": GRANT_BODY,
        "created_at": "2026-09-03T08:19:32Z",
        "updated_at": "2026-09-03T08:19:32Z",
    }
    comment.update(overrides)
    return comment


def _owner_comment(body: object, *, comment_id: int = 5600000001) -> dict[str, object]:
    return {
        "id": comment_id,
        "user": {"login": "fol2", "id": 105634418},
        "author_association": "OWNER",
        "issue_url": ISSUE_URL,
        "body": body,
        "created_at": "2026-09-04T00:00:00Z",
        "updated_at": "2026-09-04T00:00:00Z",
    }


def _ready_packet(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    proving, unpublished, connection = _stores(root)
    _nonterminal_obligation(
        connection,
        ledger_seq=1,
        item_key="candidate",
        ingest_id="candidate-ingest",
    )
    connection.commit()
    connection.close()
    _seed_proving_accountability(proving)
    authority = _authority_store(root)
    packet = _bounded_candidate_packet(proving, unpublished, authority, monkeypatch)
    assert packet["verdict"] == "READY_FOR_OWNER_DECISION"
    assert validate_graphiti_campaign_packet(packet) == packet["bounded_campaign"]
    campaign = packet["bounded_campaign"]
    assert campaign["campaign_authorised"] is False
    assert campaign["cohort"]["dispatch_authorised"] is False
    assert campaign["cohort"]["claim_performed"] is False
    return packet


def _write_packet(directory: Path, packet: dict[str, object]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "campaign-packet.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def _hooks(
    comments: CommentView,
    dispatch: DispatchStub,
    **overrides: object,
) -> dict[str, object]:
    hooks: dict[str, object] = {
        "list_comments": comments,
        "incomplete_view": lambda: False,
        "code_identity": lambda: ("head", "tree"),
        "dispatch": dispatch,
        "prior_consumption": lambda: (0, 0),
    }
    hooks.update(overrides)
    return hooks


def _run(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
    **hooks: object,
) -> tuple[int, dict[str, object]]:
    code = executor.main(argv, **hooks)
    output = capsys.readouterr().out.strip()
    payload = json.loads(output.splitlines()[-1]) if output else {}
    return code, payload


def test_authentic_grant_bytes_digest() -> None:
    assert hashlib.sha256(GRANT_BODY.encode("utf-8")).hexdigest() == GRANT_BODY_DIGEST
    assert executor.STANDING_GRANT_BYTES_DIGEST == GRANT_BODY_DIGEST


def test_spent_packet_digest_constants_include_historical_bindings() -> None:
    assert executor.SPENT_PACKET_DIGESTS == {
        "sha256:72f2f504ea465be3d7842515dd12a30828aa736f8205a920d885d8b45c90474d",
        "sha256:ecf8672ddf9cdd64c471ad68dca49d02eda673a59d86132c411632f9ec17dc04",
        "sha256:e9f5502d19c02eedbf15091eb0c3383a79006d626926cf00c1bdfa2c04229677",
        "sha256:af152f8e777b0deaf8d756d54442267577d5c902d7997f4def476c74a63be70c",
        "sha256:ffcce3ff377c231540892939a9f3a104dea143006e54499395e0396b291f7a3d",
        "sha256:7b3dd53925e3fd85b9fec85b6699cc83310a10b3d5c2fa87229cc92dc147e956",
        "sha256:8db610dbb06da896de504e9aa6de5bf5dc68769c38f3cc02b1c886a4dc5cfb40",
    }
    assert executor.DEFAULT_ATTEMPTED_EVENT_IDS >= {
        "sha256:1349b0c9f873c4da795732a974aab7db31b87cbeca50364f487f388093024cd1",
        "sha256:944101d5154cad91c2ace3629eae95b7dd893693cfa36e3e0112d8a0c3dd14da",
        "sha256:e18f0abe3fe23950c102b9ba1de2fc8db0b2957729880c08a387752c415225cc",
        "sha256:a1798f664a26148480d842a738d1d7617b59e065f6a9853d4dab3d99dffd7435",
        "sha256:8fab380a519abfe7e9feceb5165afb1dd80edd2175841c508445af8f92fe6005",
        "sha256:ac29d8d72d287b5642203c33aaff93bba0374057b3768fd4f7fa2d1c413b3f4e",
        "sha256:94c81bf8a5285df858eb0d766fe646c4c837e2c5be521a5c454fe23fea029d81",
        "sha256:c5ed75e4bbc33f702a9d730e0a9ba7542cca591ad3eaffb0bcd2145b37c4d9fa",
        "sha256:5d98022f37be76383cde3c4a049385f2bb9f108bf8113038c2d0703fd2e1ceeb",
    }


@pytest.mark.parametrize("with_release", [False, True])
def test_a_valid_fresh_binding_preflight_then_one_stubbed_dispatch(
    with_release: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView(
        [
            _grant(),
            {
                "id": 1,
                "user": {"login": "reviewer", "id": 2},
                "author_association": "CONTRIBUTOR",
                "issue_url": ISSUE_URL,
                "body": "REVOKES_COMMENT_5522755269",
                "created_at": "2026-09-04T01:00:00Z",
                "updated_at": "2026-09-04T01:00:00Z",
            },
            _owner_comment(PROGRESS_BODY),
        ]
    )
    if with_release:
        comments.comments.extend(_restriction_release())
    dispatch = DispatchStub()
    hooks = _hooks(comments, dispatch)
    argv = [
        "--packet",
        str(packet_path),
        "--evidence-root",
        str(evidence),
    ]

    preflight_code, preflight = _run(argv, capsys, **hooks)
    marker = executor.invocation_marker_path(evidence, str(packet["packet_digest"]))
    output = executor.campaign_output_dir(evidence, str(packet["packet_digest"]))
    assert preflight_code == 0
    assert preflight["event"] == "ISSUE_895_F4_PREFLIGHT_READY"
    assert preflight["effect_performed"] is False
    assert preflight["packet_digest"] == packet["packet_digest"]
    assert preflight["head_sha"] == "head"
    assert preflight["tree_sha"] == "tree"
    assert preflight["dispatch_reached"] is False
    assert preflight["invocation_marker_created"] is False
    assert dispatch.count == 0
    assert not marker.exists()
    assert not output.exists()

    dispatch_code, dispatched = _run([*argv, "--dispatch"], capsys, **hooks)
    campaign = packet["bounded_campaign"]
    assert dispatch_code == 0
    assert dispatched["dispatch_reached"] is True
    assert dispatched["dispatch_count"] == 1
    assert dispatched["invocation_marker_created"] is True
    assert dispatch.count == 1
    assert marker.is_file()
    assert stat_mode(marker) == 0o400
    assert output.is_dir()
    assert campaign["campaign_authorised"] is False
    assert campaign["cohort"]["dispatch_authorised"] is False
    assert campaign["cohort"]["claim_performed"] is False
    reloaded = json.loads(packet_path.read_text(encoding="utf-8"))
    assert reloaded["bounded_campaign"]["campaign_authorised"] is False
    assert reloaded["bounded_campaign"]["cohort"]["dispatch_authorised"] is False
    assert reloaded["bounded_campaign"]["cohort"]["claim_performed"] is False


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_b_stop_then_later_progress_refuses_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView(
        [
            _grant(),
            _owner_comment("NEWSROOM_SIGNED_STOP", comment_id=5600000002),
            _owner_comment(PROGRESS_BODY, comment_id=5600000003),
        ]
    )
    dispatch = DispatchStub()
    code, payload = _run(
        [
            "--packet",
            str(packet_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        **_hooks(comments, dispatch),
    )
    assert code == 2
    assert payload["dispatch_reached"] is False
    assert dispatch.count == 0
    assert "later restriction or signed stop is in force" in payload["message"]
    assert not executor.invocation_marker_path(
        evidence, str(packet["packet_digest"])
    ).exists()


def test_b_fence_rechecks_standing_grant_after_progress_cannot_mask_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])

    def mutate(_packet) -> None:
        comments.comments.extend(
            [
                _owner_comment("NEWSROOM_SIGNED_STOP", comment_id=5600000090),
                _owner_comment(PROGRESS_BODY, comment_id=5600000091),
            ]
        )

    dispatch = DispatchStub(before_fence=mutate)
    code, payload = _run(
        [
            "--packet",
            str(packet_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        **_hooks(comments, dispatch),
    )
    assert code == 2
    assert dispatch.count == 1
    assert payload["dispatch_reached"] is True
    assert "later restriction or signed stop is in force" in payload["message"]


@pytest.mark.parametrize(
    ("comments", "incomplete_view", "message"),
    (
        ([_owner_comment(PROGRESS_BODY)], False, "standing grant missing"),
        (
            [_grant(body=GRANT_BODY + "\n")],
            False,
            "standing grant bytes drifted",
        ),
        (
            [_grant(updated_at="2026-09-03T08:20:00Z")],
            False,
            "standing grant bytes drifted",
        ),
        (
            [_grant(user={"login": "other", "id": 105634418})],
            False,
            "standing grant missing",
        ),
        (
            [
                _grant(
                    issue_url="https://api.github.com/repos/other/newsroom/issues/895"
                )
            ],
            False,
            "standing grant missing",
        ),
        (
            [_grant(user={"login": "fol2"})],
            False,
            "authority view is incomplete",
        ),
        (
            [_grant(), _owner_comment(PROGRESS_BODY)],
            True,
            "authority view is incomplete",
        ),
        (
            [_grant(), _owner_comment('{"authority_action": "escalate"}')],
            False,
            "authority content is unresolved",
        ),
        (
            [_grant(), _owner_comment(None)],
            False,
            "authority content is unresolved",
        ),
        (
            [_grant(), _owner_comment("RESTRICTS_COMMENT_5522755269")],
            False,
            "later restriction or signed stop is in force",
        ),
        (
            [
                _grant(),
                _owner_comment(json.dumps({"authority_action": "revoke"})),
                _owner_comment(PROGRESS_BODY, comment_id=5600000004),
            ],
            False,
            "later restriction or signed stop is in force",
        ),
    ),
)
def test_b_standing_authority_refusals_never_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    comments: list[dict[str, object]],
    incomplete_view: bool,
    message: str,
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    dispatch = DispatchStub()
    for comment in comments:
        if comment.get("body") == GRANT_BODY:
            assert (
                hashlib.sha256(GRANT_BODY.encode("utf-8")).hexdigest()
                == GRANT_BODY_DIGEST
            )
    code, payload = _run(
        [
            "--packet",
            str(packet_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        **_hooks(
            CommentView(comments),
            dispatch,
            incomplete_view=lambda: incomplete_view,
        ),
    )
    assert code == 2
    assert dispatch.count == 0
    assert payload["dispatch_reached"] is False
    assert message in payload["message"]


def test_c_identity_consumption_and_lock_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])
    digest = str(packet["packet_digest"])
    event_id = packet["bounded_campaign"]["cohort"]["event_ids"][0]
    argv = [
        "--packet",
        str(packet_path),
        "--evidence-root",
        str(evidence),
        "--dispatch",
    ]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, code_identity=lambda: ("other", "tree")),
    )
    assert code == 2 and dispatch.count == 0
    assert "campaign code identity drifted" in payload["message"]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, executor_digest=lambda: "00" * 32),
    )
    assert code == 2 and dispatch.count == 0
    assert "executor identity drifted" in payload["message"]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, spent_digests={digest}),
    )
    assert code == 2 and dispatch.count == 0
    assert "packet digest is spent" in payload["message"]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, attempted_ids={event_id}),
    )
    assert code == 2 and dispatch.count == 0
    assert "attempted event is excluded" in payload["message"]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, prior_consumption=lambda: (218, 0)),
    )
    assert code == 2 and dispatch.count == 0
    assert "cumulative event allowance exhausted" in payload["message"]

    dispatch = DispatchStub()
    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, dispatch, prior_consumption=lambda: (0, 109_000_000)),
    )
    assert code == 2 and dispatch.count == 0
    assert "cumulative spend allowance exhausted" in payload["message"]

    prior_marker = evidence / f".issue-895-f4-invocation-{'ab' * 32}.json"
    prior_marker.write_text("{}\n", encoding="utf-8")
    existing = executor.invocation_marker_path(evidence, digest)
    existing.write_text("{}\n", encoding="utf-8")
    dispatch = DispatchStub()
    code, payload = _run(argv, capsys, **_hooks(comments, dispatch))
    assert code == 2 and dispatch.count == 0
    assert "invocation marker exists" in payload["message"]
    assert prior_marker.is_file()
    existing.unlink()

    output = executor.campaign_output_dir(evidence, digest)
    output.mkdir()
    dispatch = DispatchStub()
    code, payload = _run(argv, capsys, **_hooks(comments, dispatch))
    assert code == 2 and dispatch.count == 0
    assert "crash window is ambiguous" in payload["message"]
    assert prior_marker.is_file()
    output.rmdir()

    lock_path = evidence / executor.PROGRAMME_LOCK_NAME
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,os,sys;"
                "fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600);"
                "os.write(fd,b'newsroom.issue-895-standing-f4-programme-lock.v1\\n');"
                "fcntl.flock(fd,fcntl.LOCK_EX);"
                "print('LOCKED',flush=True);"
                "sys.stdin.readline();"
                "fcntl.flock(fd,fcntl.LOCK_UN);"
                "os.close(fd)"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "LOCKED"
        dispatch = DispatchStub()
        code, payload = _run(argv, capsys, **_hooks(comments, dispatch))
        assert code == 2 and dispatch.count == 0
        assert "programme lock is held" in payload["message"]
        assert prior_marker.is_file()
    finally:
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.close()
        child.wait(timeout=5)


def test_d_two_valid_packets_share_grant_and_do_not_reset_ceilings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = _ready_packet(tmp_path / "stores-a", monkeypatch)
    second = _ready_packet(tmp_path / "stores-b", monkeypatch)
    assert first["packet_digest"] != second["packet_digest"]
    first_path = _write_packet(tmp_path / "packet-a", first)
    second_path = _write_packet(tmp_path / "packet-b", second)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])
    dispatch = DispatchStub()
    cohort_len = len(first["bounded_campaign"]["cohort"]["events"])
    spend = first["bounded_campaign"]["caps"]["total"]["spend_gbp_microunits"]

    def prior_consumption() -> tuple[int, int]:
        markers = list(evidence.glob(".issue-895-f4-invocation-*.json"))
        return len(markers) * cohort_len, len(markers) * int(spend)

    hooks = _hooks(comments, dispatch, prior_consumption=prior_consumption)
    first_code, first_payload = _run(
        [
            "--packet",
            str(first_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        **hooks,
    )
    marker1 = executor.invocation_marker_path(evidence, str(first["packet_digest"]))
    assert first_code == 0
    assert first_payload["dispatch_count"] == 1
    assert dispatch.count == 1
    assert marker1.is_file()

    second_code, second_payload = _run(
        [
            "--packet",
            str(second_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        **hooks,
    )
    marker2 = executor.invocation_marker_path(evidence, str(second["packet_digest"]))
    assert second_code == 0
    assert second_payload["dispatch_count"] == 1
    assert dispatch.count == 2
    assert marker1.is_file()
    assert marker2.is_file()
    assert marker1 != marker2
    assert first["bounded_campaign"]["campaign_authorised"] is False
    assert second["bounded_campaign"]["campaign_authorised"] is False
    first_marker = json.loads(marker1.read_text(encoding="utf-8"))
    assert first_marker["event_count"] == cohort_len
    assert first_marker["spend_gbp_microunits"] == spend
    assert first_marker["invocation_record_only"] is True
    starts, reserved = executor.default_prior_consumption(evidence)
    assert starts == executor.CLASSIFIED_PRIOR_STARTS + 2 * cohort_len
    assert reserved == executor.CLASSIFIED_PRIOR_RESERVED + 2 * int(spend)


def test_default_prior_consumption_keeps_classified_baseline(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    assert executor.CLASSIFIED_PRIOR_STARTS == 9
    assert executor.CLASSIFIED_PRIOR_RESERVED == 4_000_000
    assert executor.default_prior_consumption(evidence) == (
        executor.CLASSIFIED_PRIOR_STARTS,
        executor.CLASSIFIED_PRIOR_RESERVED,
    )
    spent = evidence / (
        ".issue-895-f4-invocation-"
        "72f2f504ea465be3d7842515dd12a30828aa736f8205a920d885d8b45c90474d.json"
    )
    spent.write_text(
        json.dumps({"event_count": 214, "spend_gbp_microunits": 107000000}),
        encoding="utf-8",
    )
    successor_spent = evidence / (
        ".issue-895-f4-invocation-"
        "e9f5502d19c02eedbf15091eb0c3383a79006d626926cf00c1bdfa2c04229677.json"
    )
    successor_spent.write_text(
        json.dumps({"event_count": 212, "spend_gbp_microunits": 106000000}),
        encoding="utf-8",
    )
    extra_required_spent = evidence / (
        ".issue-895-f4-invocation-"
        "af152f8e777b0deaf8d756d54442267577d5c902d7997f4def476c74a63be70c.json"
    )
    extra_required_spent.write_text(
        json.dumps({"event_count": 211, "spend_gbp_microunits": 105500000}),
        encoding="utf-8",
    )
    dispatched_spent = evidence / (
        ".issue-895-f4-invocation-"
        "ffcce3ff377c231540892939a9f3a104dea143006e54499395e0396b291f7a3d.json"
    )
    dispatched_spent.write_text(
        json.dumps({"event_count": 211, "spend_gbp_microunits": 105500000}),
        encoding="utf-8",
    )
    persistence_spent = evidence / (
        ".issue-895-f4-invocation-"
        "7b3dd53925e3fd85b9fec85b6699cc83310a10b3d5c2fa87229cc92dc147e956.json"
    )
    persistence_spent.write_text(
        json.dumps({"event_count": 210, "spend_gbp_microunits": 105000000}),
        encoding="utf-8",
    )
    configuration_spent = evidence / (
        ".issue-895-f4-invocation-"
        "8db610dbb06da896de504e9aa6de5bf5dc68769c38f3cc02b1c886a4dc5cfb40.json"
    )
    configuration_spent.write_text(
        json.dumps({"event_count": 209, "spend_gbp_microunits": 104500000}),
        encoding="utf-8",
    )
    assert executor.default_prior_consumption(evidence) == (
        executor.CLASSIFIED_PRIOR_STARTS,
        executor.CLASSIFIED_PRIOR_RESERVED,
    )
    successor = evidence / f".issue-895-f4-invocation-{'cd' * 32}.json"
    successor.write_text(
        json.dumps({"event_count": 3, "spend_gbp_microunits": 400000}),
        encoding="utf-8",
    )
    assert executor.default_prior_consumption(evidence) == (
        executor.CLASSIFIED_PRIOR_STARTS + 3,
        executor.CLASSIFIED_PRIOR_RESERVED + 400000,
    )


def test_production_dispatch_without_runtime_refuses_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])
    captured: dict[str, object] = {}

    def fake_worker_main(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        pytest.fail("unconfigured production dispatch reached the worker")

    monkeypatch.setattr(
        "scripts.hermes_graphiti_worker.main",
        fake_worker_main,
    )
    code, payload = _run(
        [
            "--packet",
            str(packet_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        list_comments=comments,
        incomplete_view=lambda: False,
        code_identity=lambda: ("head", "tree"),
        prior_consumption=lambda: (0, 0),
    )
    marker = executor.invocation_marker_path(evidence, str(packet["packet_digest"]))
    assert code == 2
    assert payload["dispatch_reached"] is False
    assert payload["invocation_marker_created"] is False
    assert "campaign authority is unconfigured" in payload["message"]
    assert not marker.exists()
    assert "argv" not in captured


def test_production_dispatch_injects_fence_into_worker_before_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])
    captured: dict[str, object] = {}
    runtime = object()

    def fake_worker_main(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        fence = kwargs["owner_f4_fence"]
        fence(packet)
        return 0

    import scripts.hermes_graphiti_worker as worker

    monkeypatch.setattr(worker, "main", fake_worker_main)
    code, payload = _run(
        [
            "--packet",
            str(packet_path),
            "--evidence-root",
            str(evidence),
            "--dispatch",
        ],
        capsys,
        list_comments=comments,
        incomplete_view=lambda: False,
        code_identity=lambda: ("head", "tree"),
        prior_consumption=lambda: (0, 0),
        runtime=runtime,
    )
    assert code == 0
    assert payload["dispatch_reached"] is True
    assert captured["kwargs"]["runtime"] is runtime
    assert callable(captured["kwargs"]["owner_f4_fence"])
    assert captured["argv"][0] == "--campaign-packet"


def test_lock_identity_drift_and_store_attempted_and_worker_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    packet = _ready_packet(tmp_path / "stores", monkeypatch)
    packet_path = _write_packet(tmp_path / "packet", packet)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    comments = CommentView([_grant(), _owner_comment(PROGRESS_BODY)])
    argv = [
        "--packet",
        str(packet_path),
        "--evidence-root",
        str(evidence),
        "--dispatch",
    ]
    lock_path = evidence / executor.PROGRAMME_LOCK_NAME
    lock_path.write_text("other-lock-identity\n", encoding="utf-8")
    dispatch = DispatchStub()
    code, payload = _run(argv, capsys, **_hooks(comments, dispatch))
    assert code == 2 and dispatch.count == 0
    assert "programme lock identity drifted" in payload["message"]
    lock_path.unlink()

    unpublished = Path(str(packet["store_snapshots"]["unpublished"]["source_path"]))
    event_id = packet["bounded_campaign"]["cohort"]["event_ids"][0]
    connection = sqlite3.connect(unpublished)
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events "
        "SET attempt_count=1 WHERE event_id=?",
        (event_id,),
    )
    connection.commit()
    connection.close()
    dispatch = DispatchStub()
    code, payload = _run(argv, capsys, **_hooks(comments, dispatch))
    assert code == 2 and dispatch.count == 0
    assert "attempted event is excluded" in payload["message"]
    connection = sqlite3.connect(unpublished)
    connection.execute(
        "UPDATE unpublished_graphiti_revision_events "
        "SET attempt_count=0 WHERE event_id=?",
        (event_id,),
    )
    connection.commit()
    connection.close()

    def worker_stop(packet, owner_f4_fence, proving, unpublished):
        owner_f4_fence(packet)
        return 2

    code, payload = _run(
        argv,
        capsys,
        **_hooks(comments, worker_stop),
    )
    assert code == 2
    assert "campaign worker stopped" in payload["message"]


def _restriction_release():
    restriction = _owner_comment(
        json.dumps({'authority_action': 'restrict', 'scope': 'old-code sealer'}),
        comment_id=5600000100,
    )
    release = _owner_comment(json.dumps({
        'authority_action': 'release_restriction',
        'restriction_comment_id': restriction['id'],
        'restriction_body_sha256': hashlib.sha256(restriction['body'].encode()).hexdigest(),
        'standing_grant_reference': executor.STANDING_GRANT_ID,
    }), comment_id=5600000101)
    return restriction, release


def test_exact_owner_release_resolves_only_named_restriction():
    restriction, release = _restriction_release()
    # Input order is not an authority signal; the durable comment IDs are.
    executor.verify_standing_grant(CommentView([release, _grant(), restriction]))
    with pytest.raises(executor.GraphitiCampaignStop, match='signed stop'):
        executor.verify_standing_grant(CommentView([
            _grant(), restriction, release,
            _owner_comment('NEWSROOM_SIGNED_STOP', comment_id=5600000102),
        ]))
    other = _owner_comment(json.dumps({'authority_action': 'restrict'}), comment_id=5600000102)
    with pytest.raises(executor.GraphitiCampaignStop, match='signed stop'):
        executor.verify_standing_grant(CommentView([_grant(), restriction, release, other]))


@pytest.mark.parametrize('mutation', ['owner', 'issue', 'earlier', 'digest', 'grant', 'target_missing', 'release_edited', 'target_edited', 'target_revoke', 'target_signed_stop'])
def test_restriction_release_refuses_inexact_authority(mutation):
    restriction, release = _restriction_release()
    body = json.loads(release['body'])
    if mutation == 'owner':
        release['user'] = {'login': 'other', 'id': 1}
    elif mutation == 'issue':
        release['issue_url'] = ISSUE_URL + '0'
    elif mutation == 'release_edited':
        release['updated_at'] = '2026-09-08T00:00:00Z'
    elif mutation == 'earlier':
        release['id'] = restriction['id'] - 1
    elif mutation in ('digest', 'grant', 'target_missing'):
        field, value = {'digest': ('restriction_body_sha256', '0' * 64), 'grant': ('standing_grant_reference', 1), 'target_missing': ('restriction_comment_id', 1)}[mutation]
        body[field] = value
    else:
        restriction['body'] = {
            'target_edited': json.dumps({'authority_action': 'restrict', 'scope': 'changed'}),
            'target_revoke': json.dumps({'authority_action': 'revoke'}),
            'target_signed_stop': json.dumps({'authority_action': 'restrict', 'reason': 'NEWSROOM_SIGNED_STOP'}),
        }[mutation]
        if mutation != 'target_edited':
            body['restriction_body_sha256'] = hashlib.sha256(restriction['body'].encode()).hexdigest()
    release['body'] = json.dumps(body)
    with pytest.raises(executor.GraphitiCampaignStop):
        executor.verify_standing_grant(CommentView([_grant(), restriction, release]))
