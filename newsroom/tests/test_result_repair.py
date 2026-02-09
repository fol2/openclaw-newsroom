import unittest

from newsroom.result_repair import repair_result_json
from newsroom.validators.news_reporter_script_v1 import _count_cjk


class TestResultRepair(unittest.TestCase):
    def test_repairs_read_more_urls_to_meet_min_and_include_primary(self) -> None:
        job = {
            "story": {
                "title": "【測試】港股消息",
                "primary_url": "https://example.com/a",
                "supporting_urls": [
                    "https://example.net/b",
                    "https://example.org/c",
                ],
            },
            "state": {
                "source_pack": {
                    "sources": [
                        {"url": "https://example.com/a", "on_topic": True, "selected_chars": 800},
                        {"url": "https://example.net/b", "on_topic": True, "selected_chars": 800},
                    ]
                }
            },
        }
        result_json = {
            "status": "SUCCESS",
            "title": "English Title",
            "draft": {"body": "x", "read_more_urls": ["https://example.net/b"]},
            "sources_used": ["https://example.net/b"],
        }
        errors = [
            "success_requires:read_more_urls_3_to_5",
            "success_requires:read_more_includes_primary_url",
        ]

        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=errors)
        self.assertIn("read_more_urls", repairs)
        self.assertIn("https://example.com/a", repaired["draft"]["read_more_urls"])
        self.assertGreaterEqual(len(repaired["draft"]["read_more_urls"]), 3)

    def test_repairs_title_from_job_when_validator_requires_traditional_chinese(self) -> None:
        job = {"story": {"title": "【AI】新模型發布"}}
        result_json = {"status": "SUCCESS", "title": "English only", "draft": {"body": "x", "read_more_urls": []}, "sources_used": []}
        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=["success_requires:title_traditional_chinese"])
        self.assertIn("title_from_job", repairs)
        self.assertEqual(repaired["title"], "【AI】新模型發布")

    def test_repairs_sources_used_min_two(self) -> None:
        job = {
            "story": {"primary_url": "https://example.com/a"},
            "state": {
                "source_pack": {
                    "sources": [
                        {"url": "https://example.com/a", "on_topic": True},
                        {"url": "https://example.net/b", "on_topic": True},
                    ]
                }
            },
        }
        result_json = {
            "status": "SUCCESS",
            "title": "【測試】",
            "draft": {"body": "x", "read_more_urls": ["https://example.com/a", "https://example.net/b", "https://example.org/c"]},
            "sources_used": ["https://example.com/a"],
        }
        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=["success_requires:sources_used_min_2"])
        self.assertIn("sources_used", repairs)
        self.assertGreaterEqual(len(repaired["sources_used"]), 2)

    def test_pads_body_cjk_length_for_small_underflow(self) -> None:
        job = {"story": {"title": "【測試】"}}
        body = "中" * 2699
        result_json = {"status": "SUCCESS", "title": "【測試】", "draft": {"body": body, "read_more_urls": []}, "sources_used": []}
        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=["success_requires:body_cjk_2700_to_3300"])
        self.assertIn("body_cjk_pad", repairs)
        self.assertGreaterEqual(len(repaired["draft"]["body"]), len(body))

    def test_expands_body_cjk_length_for_moderate_underflow(self) -> None:
        job = {"story": {"title": "【測試】"}}
        body = "中" * 2200
        result_json = {"status": "SUCCESS", "title": "【測試】", "draft": {"body": body, "read_more_urls": []}, "sources_used": []}
        repaired, repairs = repair_result_json(result_json=result_json, job=job, errors=["success_requires:body_cjk_2700_to_3300"])
        self.assertIn("body_cjk_expand", repairs)
        cjk = _count_cjk(repaired["draft"]["body"])
        self.assertGreaterEqual(cjk, 2700)
        self.assertLessEqual(cjk, 3300)


if __name__ == "__main__":
    unittest.main()
