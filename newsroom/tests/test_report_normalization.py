import unittest

from newsroom.runner import _normalize_report_body  # type: ignore[attr-defined]


class TestReportNormalization(unittest.TestCase):
    def test_strips_numbered_section_headings_plain(self) -> None:
        body = "\n".join(
            [
                "（截至英國時間 2026-02-04 15:58）",
                "",
                "1) 背景脈絡：第一段",
                "第二行",
                "",
                "2) 今次發生咩事",
                "內容A",
                "",
                "6) 後續可能點走：內容B",
            ]
        )
        out = _normalize_report_body(body)
        self.assertIn("【背景脈絡】\n第一段", out)
        self.assertIn("【今次發生咩事】", out)
        self.assertIn("【後續可能點走】\n內容B", out)
        self.assertNotIn("1)", out)
        self.assertNotIn("2)", out)
        self.assertNotIn("6)", out)

    def test_strips_numbered_section_headings_bracketed(self) -> None:
        body = "\n".join(
            [
                "1)【背景脈絡】 第一段",
                "2) 【今次發生咩事】：第二段",
                "3) 【關鍵人物／機構】",
                "內容C",
            ]
        )
        out = _normalize_report_body(body)
        self.assertIn("【背景脈絡】\n第一段", out)
        self.assertIn("【今次發生咩事】\n第二段", out)
        self.assertIn("【關鍵人物／機構】", out)
        self.assertNotIn("1)", out)
        self.assertNotIn("2)", out)
        self.assertNotIn("3)", out)

    def test_does_not_rewrite_unrelated_numbered_lines(self) -> None:
        body = "1) 2026 年有咩大事？\\n2) 另一個問題"
        out = _normalize_report_body(body)
        self.assertEqual(out, body)

