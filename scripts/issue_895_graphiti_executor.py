#!/usr/bin/env python3
"""Provider-free #895 successor-binding executor; default mode is preflight."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from newsroom.control_plane.graphiti_steady_state import (
    validate_graphiti_campaign_packet,
)
from scripts.hermes_graphiti_worker import GraphitiCampaignStop

STANDING_GRANT_ID = 5522755269
STANDING_GRANT_BYTES_DIGEST = (
    "ab068ff3c70bc8a09254357026c89ddda64af2a26f4e951513d8bbfde3092459"
)
OWNER_LOGIN = "fol2"
OWNER_USER_ID = 105634418
ISSUE_URL_MARK = "/repos/fol2/newsroom/issues/895"
PROGRAMME_LOCK_NAME = ".issue-895-f4-programme.lock"
PROGRAMME_LOCK_IDENTITY = "newsroom.issue-895-standing-f4-programme-lock.v1\n"
PROGRAMME_MAX_EVENTS = 218
PROGRAMME_MAX_SPEND = 109_000_000
CLASSIFIED_PRIOR_STARTS = 9
CLASSIFIED_PRIOR_RESERVED = 4_000_000
SPENT_PACKET_DIGESTS = frozenset(
    {
        "sha256:72f2f504ea465be3d7842515dd12a30828aa736f8205a920d885d8b45c90474d",
        "sha256:ecf8672ddf9cdd64c471ad68dca49d02eda673a59d86132c411632f9ec17dc04",
        "sha256:e9f5502d19c02eedbf15091eb0c3383a79006d626926cf00c1bdfa2c04229677",
        "sha256:af152f8e777b0deaf8d756d54442267577d5c902d7997f4def476c74a63be70c",
        "sha256:ffcce3ff377c231540892939a9f3a104dea143006e54499395e0396b291f7a3d",
        "sha256:7b3dd53925e3fd85b9fec85b6699cc83310a10b3d5c2fa87229cc92dc147e956",
        "sha256:8db610dbb06da896de504e9aa6de5bf5dc68769c38f3cc02b1c886a4dc5cfb40",
    }
)
DEFAULT_ATTEMPTED_EVENT_IDS = frozenset(
    {
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
)
RESTRICTION_TOKENS = (
    "NEWSROOM_SIGNED_STOP",
    "NEWSROOM_OWNER_RESTRICTION",
    "REVOKES_COMMENT_5522755269",
    "RESTRICTS_COMMENT_5522755269",
)
AUTHORITY_STOP_ACTIONS = frozenset({"stop", "revoke", "restrict"})
REQUIRED_COMMENT_FIELDS = ("id", "user", "issue_url", "body", "author_association")
TERMINAL_EVIDENCE_NAME = "terminal-evidence-manifest.json"

ListComments = Callable[[], Sequence[Mapping[str, object]]]
CodeIdentity = Callable[[], tuple[str, str]]
Dispatch = Callable[
    [Mapping[str, object], Callable[[Mapping[str, object]], None], str, str],
    object,
]


def stop(message: str) -> None:
    raise GraphitiCampaignStop(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def executor_source_digest() -> str:
    return _sha256_bytes(Path(__file__).resolve().read_bytes())


def invocation_marker_path(evidence_root: Path, packet_digest: str) -> Path:
    return evidence_root / (
        f".issue-895-f4-invocation-{packet_digest.removeprefix('sha256:')}.json"
    )


def campaign_output_dir(evidence_root: Path, packet_digest: str) -> Path:
    return evidence_root / f"f4-campaign-{packet_digest.removeprefix('sha256:')}"


def default_code_identity() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


def default_list_comments() -> list[Mapping[str, object]]:
    try:
        raw = subprocess.run(
            (
                "gh",
                "api",
                "--paginate",
                "repos/fol2/newsroom/issues/895/comments?per_page=100",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        values = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise GraphitiCampaignStop("authority view is incomplete") from exc
    if not isinstance(values, list):
        stop("authority view is incomplete")
    return values


def _write_exclusive(path: Path, payload: object, *, mode: int = 0o400) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _copy_exclusive(source: Path, destination: Path, *, mode: int = 0o400) -> None:
    payload = source.read_bytes()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _terminal_evidence_complete(output: Path) -> bool:
    evidence = output / TERMINAL_EVIDENCE_NAME
    return evidence.is_file() and evidence.stat().st_size > 0


def default_prior_consumption(evidence_root: Path) -> tuple[int, int]:
    """Retained classified starts/reservations, plus non-spent successor markers."""

    starts = CLASSIFIED_PRIOR_STARTS
    reserved = CLASSIFIED_PRIOR_RESERVED
    try:
        markers = sorted(evidence_root.glob(".issue-895-f4-invocation-*.json"))
    except OSError as exc:
        raise GraphitiCampaignStop("cannot inspect programme consumption") from exc
    for marker in markers:
        digest = "sha256:" + marker.name.removeprefix(
            ".issue-895-f4-invocation-"
        ).removesuffix(".json")
        if digest in SPENT_PACKET_DIGESTS:
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphitiCampaignStop("cannot inspect programme consumption") from exc
        if not isinstance(payload, Mapping):
            stop("cannot inspect programme consumption")
        event_count = payload.get("event_count")
        spend = payload.get("spend_gbp_microunits")
        if isinstance(event_count, bool) or not isinstance(event_count, int):
            stop("cannot inspect programme consumption")
        if isinstance(spend, bool) or not isinstance(spend, int):
            stop("cannot inspect programme consumption")
        if event_count < 0 or spend < 0:
            stop("cannot inspect programme consumption")
        starts += event_count
        reserved += spend
    return starts, reserved


def unpublished_attempted_event_ids(unpublished_path: str) -> frozenset[str]:
    try:
        connection = sqlite3.connect(f"file:{unpublished_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise GraphitiCampaignStop("cannot inspect attempted events") from exc
    try:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='unpublished_graphiti_revision_events'"
        ).fetchone()
        if present is None:
            return frozenset()
        rows = connection.execute(
            "SELECT event_id FROM unpublished_graphiti_revision_events "
            "WHERE attempt_count>0 OR provider_dispatched>0"
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)
    except sqlite3.Error as exc:
        raise GraphitiCampaignStop("cannot inspect attempted events") from exc
    finally:
        connection.close()


@contextlib.contextmanager
def programme_lock(evidence_root: Path):
    path = evidence_root / PROGRAMME_LOCK_NAME
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise GraphitiCampaignStop("programme lock is held") from exc
        handle.seek(0)
        value = handle.read()
        if not value:
            handle.write(PROGRAMME_LOCK_IDENTITY)
            handle.flush()
            os.fsync(handle.fileno())
        elif value != PROGRAMME_LOCK_IDENTITY:
            stop("programme lock identity drifted")
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        stop(f"{name} is not a mapping")
    return value


def _comment_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_owner(comment: Mapping[str, object]) -> bool:
    user = comment.get("user")
    if not isinstance(user, Mapping):
        return False
    return (
        user.get("login") == OWNER_LOGIN
        and user.get("id") == OWNER_USER_ID
        and comment.get("author_association") == "OWNER"
    )


def _raise_if_incomplete(incomplete_view: object) -> None:
    if incomplete_view is None:
        return
    if callable(incomplete_view):
        try:
            flagged = incomplete_view()
        except GraphitiCampaignStop:
            raise
        except Exception as exc:
            raise GraphitiCampaignStop("authority view is incomplete") from exc
    else:
        flagged = bool(incomplete_view)
    if flagged:
        stop("authority view is incomplete")


def _authority_action(body: str) -> str | None:
    stripped = body.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping) or "authority_action" not in value:
        return None
    action = value.get("authority_action")
    return action if isinstance(action, str) else ""


def _released_restrictions(comments: Sequence[Mapping[str, object]]) -> set[int]:
    """Resolve exact owner releases, never infer expiry from progress prose."""
    by_id = {comment.get("id"): comment for comment in comments}
    released: set[int] = set()
    for comment in comments:
        body = comment.get("body")
        if not _is_owner(comment) or not isinstance(body, str):
            continue
        if _authority_action(body) != "release_restriction":
            continue
        value = json.loads(body)
        target_id = _comment_id(value.get("restriction_comment_id"))
        target = by_id.get(target_id)
        release_id = _comment_id(comment.get("id"))
        if (
            set(value) != {"authority_action", "restriction_comment_id",
                           "restriction_body_sha256", "standing_grant_reference"}
            or value.get("standing_grant_reference") != STANDING_GRANT_ID
            or target is None or target_id is None or release_id is None
            or release_id <= target_id
            or not comment.get("created_at")
            or comment.get("created_at") != comment.get("updated_at")
            or comment.get("issue_url") != f"https://api.github.com{ISSUE_URL_MARK}"
            or target.get("issue_url") != comment.get("issue_url")
            or not _is_owner(target)
        ):
            stop("restriction release identity differs")
        target_body = target.get("body")
        if (
            not isinstance(target_body, str)
            or _authority_action(target_body) != "restrict"
            or any(token in target_body for token in RESTRICTION_TOKENS)
            or _sha256_bytes(target_body.encode("utf-8"))
            != value.get("restriction_body_sha256")
        ):
            stop("restriction release target differs")
        released.add(target_id)
    return released


def verify_standing_grant(
    list_comments: ListComments,
    incomplete_view: object = None,
) -> None:
    _raise_if_incomplete(incomplete_view)
    try:
        comments = list_comments()
    except GraphitiCampaignStop:
        raise
    except Exception as exc:
        raise GraphitiCampaignStop("authority view is incomplete") from exc
    if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes)):
        stop("authority view is incomplete")

    grant: Mapping[str, object] | None = None
    for raw in comments:
        comment = _mapping(raw, "authority comment")
        if any(field not in comment for field in REQUIRED_COMMENT_FIELDS):
            stop("authority view is incomplete")
        user = comment.get("user")
        if not isinstance(user, Mapping) or "login" not in user or "id" not in user:
            stop("authority view is incomplete")
        if _comment_id(comment.get("id")) == STANDING_GRANT_ID:
            grant = comment

    if grant is None:
        stop("standing grant missing")
    issue_url = grant.get("issue_url")
    if not isinstance(issue_url, str) or ISSUE_URL_MARK not in issue_url:
        stop("standing grant missing")
    if not _is_owner(grant):
        stop("standing grant missing")
    body = grant.get("body")
    if not isinstance(body, str):
        stop("standing grant bytes drifted")
    if _sha256_bytes(body.encode("utf-8")) != STANDING_GRANT_BYTES_DIGEST:
        stop("standing grant bytes drifted")
    if grant.get("created_at") != grant.get("updated_at"):
        stop("standing grant bytes drifted")

    released = _released_restrictions(comments)
    for raw in comments:
        comment = _mapping(raw, "authority comment")
        if _comment_id(comment.get("id")) == STANDING_GRANT_ID or not _is_owner(
            comment
        ):
            continue
        body = comment.get("body")
        if body is None or body == "":
            stop("authority content is unresolved")
        if not isinstance(body, str):
            stop("authority content is unresolved")
        action = _authority_action(body)
        if action == "release_restriction" or comment.get("id") in released:
            continue
        if action is not None and action not in AUTHORITY_STOP_ACTIONS:
            stop("authority content is unresolved")
        if action in AUTHORITY_STOP_ACTIONS or any(
            token in body for token in RESTRICTION_TOKENS
        ):
            stop("later restriction or signed stop is in force")


def verify_code_identity(
    packet: Mapping[str, object],
    code_identity: CodeIdentity,
) -> tuple[str, str]:
    head_sha, tree_sha = code_identity()
    expected = packet.get("code_identity")
    if expected != {"head_sha": head_sha, "tree_sha": tree_sha}:
        stop("campaign code identity drifted")
    return head_sha, tree_sha


def verify_executor_identity(executor_digest: object) -> str:
    actual = executor_source_digest()
    if executor_digest is None:
        expected = actual
    elif callable(executor_digest):
        expected = executor_digest()
    else:
        expected = str(executor_digest)
    if actual != expected:
        stop("executor identity drifted")
    return actual


def load_validated_packet(packet_path: Path) -> tuple[dict[str, object], dict[str, object], str]:
    try:
        packet_bytes = packet_path.read_bytes()
        loaded = json.loads(packet_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphitiCampaignStop("campaign packet is not a JSON object") from exc
    if not isinstance(loaded, dict):
        stop("campaign packet is not a JSON object")
    try:
        campaign = validate_graphiti_campaign_packet(loaded)
    except (TypeError, ValueError) as exc:
        raise GraphitiCampaignStop(str(exc)) from exc
    digest = loaded.get("packet_digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        stop("campaign packet canonical digest differs")
    return loaded, campaign, digest


def verify_consumption(
    *,
    campaign: Mapping[str, object],
    packet_digest: str,
    evidence_root: Path,
    prior_consumption: Callable[[], tuple[int, int]],
    attempted_ids: Iterable[str],
    spent_digests: Iterable[str],
) -> None:
    excluded_packets = SPENT_PACKET_DIGESTS | set(spent_digests)
    if packet_digest in excluded_packets:
        stop("packet digest is spent")
    cohort = _mapping(campaign.get("cohort"), "campaign cohort")
    events = cohort.get("events")
    if not isinstance(events, list) or not events:
        stop("campaign packet cohort differs")
    event_ids = {
        str(_mapping(item, "campaign event").get("event_id")) for item in events
    }
    excluded_events = DEFAULT_ATTEMPTED_EVENT_IDS | set(attempted_ids)
    if event_ids & excluded_events:
        stop("attempted event is excluded")
    caps = _mapping(_mapping(campaign.get("caps"), "caps").get("total"), "total caps")
    spend_cap = caps.get("spend_gbp_microunits")
    if isinstance(spend_cap, bool) or not isinstance(spend_cap, int):
        stop("campaign packet caps differ")
    try:
        prior_starts, prior_reserved = prior_consumption()
    except GraphitiCampaignStop:
        raise
    except Exception as exc:
        raise GraphitiCampaignStop("authority view is incomplete") from exc
    if prior_starts + len(events) > PROGRAMME_MAX_EVENTS:
        stop("cumulative event allowance exhausted")
    if prior_reserved + spend_cap > PROGRAMME_MAX_SPEND:
        stop("cumulative spend allowance exhausted")
    marker = invocation_marker_path(evidence_root, packet_digest)
    output = campaign_output_dir(evidence_root, packet_digest)
    if marker.exists():
        stop("invocation marker exists")
    if output.exists() and not _terminal_evidence_complete(output):
        stop("crash window is ambiguous")


def store_paths(packet: Mapping[str, object]) -> tuple[str, str]:
    stores = _mapping(packet.get("store_snapshots"), "store snapshots")
    proving = _mapping(stores.get("proving"), "proving snapshot")
    unpublished = _mapping(stores.get("unpublished"), "unpublished snapshot")
    proving_path = proving.get("source_path")
    unpublished_path = unpublished.get("source_path")
    if not isinstance(proving_path, str) or not isinstance(unpublished_path, str):
        stop("campaign packet store snapshots differ")
    return proving_path, unpublished_path


def bind_invocation(
    packet_path: Path,
    *,
    evidence_root: Path,
    list_comments: ListComments,
    incomplete_view: object,
    code_identity: CodeIdentity,
    executor_digest: object,
    prior_consumption: Callable[[], tuple[int, int]],
    attempted_ids: Iterable[str],
    spent_digests: Iterable[str],
) -> tuple[dict[str, object], dict[str, object], str, str, str, str]:
    packet, campaign, digest = load_validated_packet(packet_path)
    head_sha, tree_sha = verify_code_identity(packet, code_identity)
    executor_sha = verify_executor_identity(executor_digest)
    verify_standing_grant(list_comments, incomplete_view)
    _proving, unpublished = store_paths(packet)
    verify_consumption(
        campaign=campaign,
        packet_digest=digest,
        evidence_root=evidence_root,
        prior_consumption=prior_consumption,
        attempted_ids=set(attempted_ids) | unpublished_attempted_event_ids(unpublished),
        spent_digests=spent_digests,
    )
    return packet, campaign, digest, head_sha, tree_sha, executor_sha


def make_owner_f4_fence(
    *,
    packet_digest: str,
    list_comments: ListComments,
    incomplete_view: object,
    code_identity: CodeIdentity,
) -> Callable[[Mapping[str, object]], None]:
    def owner_f4_fence(supplied: Mapping[str, object]) -> None:
        if supplied.get("packet_digest") != packet_digest:
            stop("worker supplied a different packet")
        verify_code_identity(supplied, code_identity)
        verify_standing_grant(list_comments, incomplete_view)

    return owner_f4_fence


def _emit(payload: Mapping[str, object], stdout: TextIO) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True, file=stdout)


def _production_dispatch(
    packet_copy: Path,
    runtime: object | None,
) -> Dispatch:
    def dispatch(
        packet: Mapping[str, object],
        owner_f4_fence: Callable[[Mapping[str, object]], None],
        proving: str,
        unpublished: str,
    ) -> object:
        from scripts import hermes_graphiti_worker as worker

        argv = [
            "--campaign-packet",
            str(packet_copy),
            "--proving",
            proving,
            "--unpublished",
            unpublished,
        ]
        return worker.main(argv, runtime=runtime, owner_f4_fence=owner_f4_fence)

    return dispatch


def main(
    argv: Sequence[str] | None = None,
    *,
    list_comments: ListComments | None = None,
    incomplete_view: object = None,
    code_identity: CodeIdentity | None = None,
    executor_digest: object = None,
    dispatch: Dispatch | None = None,
    prior_consumption: Callable[[], tuple[int, int]] | None = None,
    attempted_ids: Iterable[str] | None = None,
    spent_digests: Iterable[str] | None = None,
    runtime: object | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dispatch", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = stdout if stdout is not None else sys.stdout
    comments = list_comments or default_list_comments
    identity = code_identity or default_code_identity
    prior = prior_consumption or (
        lambda: default_prior_consumption(args.evidence_root)
    )
    extra_attempted = () if attempted_ids is None else attempted_ids
    extra_spent = () if spent_digests is None else spent_digests
    dispatch_mode = bool(args.dispatch)
    dispatch_count = 0
    marker_created = False
    try:
        packet, campaign, digest, head_sha, tree_sha, executor_sha = bind_invocation(
            args.packet,
            evidence_root=args.evidence_root,
            list_comments=comments,
            incomplete_view=incomplete_view,
            code_identity=identity,
            executor_digest=executor_digest,
            prior_consumption=prior,
            attempted_ids=extra_attempted,
            spent_digests=extra_spent,
        )
        if not dispatch_mode:
            _emit(
                {
                    "event": "ISSUE_895_F4_PREFLIGHT_READY",
                    "effect_performed": False,
                    "packet_digest": digest,
                    "head_sha": head_sha,
                    "tree_sha": tree_sha,
                    "dispatch_reached": False,
                    "dispatch_count": 0,
                    "invocation_marker_created": False,
                },
                out,
            )
            return 0
        if dispatch is None and runtime is None:
            stop("campaign authority is unconfigured")

        fence = make_owner_f4_fence(
            packet_digest=digest,
            list_comments=comments,
            incomplete_view=incomplete_view,
            code_identity=identity,
        )
        proving, unpublished = store_paths(packet)
        output = campaign_output_dir(args.evidence_root, digest)
        marker = invocation_marker_path(args.evidence_root, digest)
        packet_copy = output / "campaign-packet.json"
        dispatch_hook = dispatch or _production_dispatch(packet_copy, runtime)
        with programme_lock(args.evidence_root):
            if marker.exists():
                stop("invocation marker exists")
            if output.exists() and not _terminal_evidence_complete(output):
                stop("crash window is ambiguous")
            try:
                output.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError as exc:
                raise GraphitiCampaignStop("crash window is ambiguous") from exc
            _copy_exclusive(args.packet, packet_copy)
            events = _mapping(campaign.get("cohort"), "campaign cohort").get("events")
            if not isinstance(events, list):
                stop("campaign packet cohort differs")
            spend_cap = _mapping(
                _mapping(campaign.get("caps"), "caps").get("total"),
                "total caps",
            ).get("spend_gbp_microunits")
            if isinstance(spend_cap, bool) or not isinstance(spend_cap, int):
                stop("campaign packet caps differ")
            _write_exclusive(
                marker,
                {
                    "packet_digest": digest,
                    "executor_sha256": executor_sha,
                    "head_sha": head_sha,
                    "tree_sha": tree_sha,
                    "event_count": len(events),
                    "spend_gbp_microunits": spend_cap,
                    "invocation_record_only": True,
                },
            )
            marker_created = True
            dispatch_count += 1
            result = dispatch_hook(packet, fence, proving, unpublished)
            if (
                isinstance(result, int)
                and not isinstance(result, bool)
                and result != 0
            ):
                stop("campaign worker stopped")
        _emit(
            {
                "event": "ISSUE_895_F4_DISPATCH_READY",
                "effect_performed": False,
                "packet_digest": digest,
                "head_sha": head_sha,
                "tree_sha": tree_sha,
                "dispatch_reached": True,
                "dispatch_count": dispatch_count,
                "invocation_marker_created": True,
                "result": result,
            },
            out,
        )
        return 0
    except GraphitiCampaignStop as exc:
        _emit(
            {
                "event": "ISSUE_895_F4_REFUSED",
                "effect_performed": False,
                "dispatch_reached": dispatch_count > 0,
                "dispatch_count": dispatch_count,
                "invocation_marker_created": marker_created,
                "message": str(exc),
            },
            out,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
