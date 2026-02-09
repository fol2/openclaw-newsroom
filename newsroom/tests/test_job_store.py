import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from newsroom.job_store import FileLock, LockHeldError


def _pid_max() -> int:
    try:
        return int(Path("/proc/sys/kernel/pid_max").read_text(encoding="utf-8").strip())
    except Exception:
        # Sensible fallback on Linux.
        return 4194304


class TestJobStoreLocks(unittest.TestCase):
    def test_file_lock_reclaims_dead_pid_without_waiting_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "job.json.lock"

            # Create a lock file that looks "fresh" by mtime, but points to a PID that cannot exist.
            dead_pid = _pid_max() + 1
            payload = {"owner": "test", "pid": dead_pid, "hostname": socket.gethostname(), "created_at": "now"}
            lock_path.write_text(json.dumps(payload), encoding="utf-8")

            lock = FileLock(lock_path, owner="test2", ttl_seconds=3600)
            lock.acquire()
            try:
                self.assertTrue(lock_path.exists())
                # The lock file should have been overwritten by this process.
                obj = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(int(obj.get("pid")), os.getpid())
            finally:
                lock.release()
            self.assertFalse(lock_path.exists())

    def test_file_lock_stale_reclaim_race_does_not_raise_fileexists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "job.json.lock"
            lock = FileLock(lock_path, owner="test", ttl_seconds=1)

            # Simulate: lock exists -> stale -> unlink -> another runner recreates lock
            # before we can acquire. We should raise LockHeldError (not leak FileExistsError).
            lock_path.write_text("{}", encoding="utf-8")

            real_os_open = os.open
            call_count = 0

            def _fake_open(path, flags, mode):  # type: ignore[no-untyped-def]
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise FileExistsError()
                if call_count == 2:
                    # Another runner recreated the lock after we unlinked it.
                    lock_path.write_text("{}", encoding="utf-8")
                    raise FileExistsError()
                return real_os_open(path, flags, mode)

            with (
                patch("newsroom.job_store.os.open", side_effect=_fake_open),
                patch("newsroom.job_store._is_lock_stale", side_effect=[True, False]),
            ):
                with self.assertRaises(LockHeldError):
                    lock.acquire()

    def test_file_lock_fileexists_race_lock_deleted_allows_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "job.json.lock"
            lock_path.write_text("{}", encoding="utf-8")

            lock = FileLock(lock_path, owner="test", ttl_seconds=3600)

            real_os_open = os.open
            call_count = 0

            def _fake_open(path, flags, mode):  # type: ignore[no-untyped-def]
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Another runner released/removed the lock between the O_EXCL check and our inspection.
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    raise FileExistsError()
                return real_os_open(path, flags, mode)

            with patch("newsroom.job_store.os.open", side_effect=_fake_open):
                lock.acquire()
                try:
                    obj = json.loads(lock_path.read_text(encoding="utf-8"))
                    self.assertEqual(int(obj.get("pid")), os.getpid())
                finally:
                    lock.release()
            self.assertFalse(lock_path.exists())
