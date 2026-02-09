from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LockHeldError(RuntimeError):
    pass


@dataclass(frozen=True)
class LockInfo:
    owner: str
    pid: int
    hostname: str
    created_at: str


def utc_iso() -> str:
    # ISO-ish string, stable and readable in logs.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_lock_info(lock_path: Path) -> LockInfo | None:
    try:
        obj = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return LockInfo(
            owner=str(obj.get("owner", "")),
            pid=int(obj.get("pid", 0)),
            hostname=str(obj.get("hostname", "")),
            created_at=str(obj.get("created_at", "")),
        )
    except Exception:
        return None


def _is_lock_stale(lock_path: Path, ttl_seconds: int) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False

    # Primary staleness signal: age-based TTL.
    if age > ttl_seconds:
        return True

    # Secondary staleness signal: the owning process no longer exists on this host.
    # This helps the cron runner recover quickly after an unclean termination (SIGKILL/OOM)
    # without having to wait for the full TTL.
    info = _read_lock_info(lock_path)
    if info and info.hostname and info.pid:
        try:
            if info.hostname == socket.gethostname():
                os.kill(int(info.pid), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Can't inspect; assume it's still alive.
            return False
        except Exception:
            return False
    return False


class FileLock:
    """O_EXCL-style lockfile.

    The lock is held for the duration of the job processing. If the runner crashes,
    the lock file remains and will be treated as stale after ttl_seconds.
    """

    def __init__(self, lock_path: Path, *, owner: str, ttl_seconds: int) -> None:
        self.lock_path = lock_path
        self.owner = owner
        self.ttl_seconds = ttl_seconds
        self.acquired = False

    def acquire(self) -> None:
        lock_dir = self.lock_path.parent
        lock_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "owner": self.owner,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": utc_iso(),
        }
        raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd: int | None = None
        for _attempt in range(2):
            try:
                fd = os.open(self.lock_path, flags, 0o644)
                break
            except FileExistsError:
                # If stale, clear it and retry. Another runner may race and recreate
                # the lock after unlink; in that case we treat it as held.
                if _is_lock_stale(self.lock_path, self.ttl_seconds):
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                info = _read_lock_info(self.lock_path)
                raise LockHeldError(f"lock held: {self.lock_path} info={info}") from None
        if fd is None:
            info = _read_lock_info(self.lock_path)
            raise LockHeldError(f"lock held: {self.lock_path} info={info}") from None

        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)

        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def jail_job_file(path: Path, *, reason: str) -> Path:
    """Quarantine a corrupt/unprocessable job file by renaming it to .jail.

    Returns the new path.  Also removes any stale .lock file and writes a
    small .reason sidecar so operators can triage later.
    """
    jail_path = path.with_suffix(path.suffix + ".jail")
    try:
        os.replace(path, jail_path)
    except FileNotFoundError:
        # Already gone (another runner jailed it first).
        return jail_path

    # Best-effort: write a reason file.
    try:
        reason_path = Path(str(jail_path) + ".reason")
        reason_path.write_text(f"{utc_iso()} {reason}\n", encoding="utf-8")
    except Exception:
        pass

    # Best-effort: remove stale lock.
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass

    return jail_path
