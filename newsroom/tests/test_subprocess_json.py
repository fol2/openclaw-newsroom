import sys
import unittest

from newsroom.subprocess_json import run_json_command


class TestRunJsonCommand(unittest.TestCase):
    def test_success_parses_json_object(self) -> None:
        cmd = [sys.executable, "-c", "print('{\"ok\": true, \"value\": 123}')"]
        res = run_json_command(cmd, timeout_seconds=2, max_output_chars=2000)
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("value"), 123)

    def test_timeout_returns_structured_error_with_truncation(self) -> None:
        # Emit a lot of output, then hang long enough to trigger the timeout.
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys, time;"
                "sys.stdout.write('a' * 5000); sys.stdout.flush();"
                "sys.stderr.write('b' * 5000); sys.stderr.flush();"
                "time.sleep(5)"
            ),
        ]
        res = run_json_command(cmd, timeout_seconds=0.5, max_output_chars=2000)
        self.assertFalse(res.get("ok"))
        self.assertEqual(res.get("error"), "timeout")
        self.assertEqual(res.get("timeout_seconds"), 0.5)
        self.assertEqual(res.get("cmd"), cmd)
        self.assertEqual(len(res.get("stdout", "")), 2000)
        self.assertEqual(len(res.get("stderr", "")), 2000)
        self.assertEqual(res.get("stdout"), "a" * 2000)
        self.assertEqual(res.get("stderr"), "b" * 2000)

