import unittest

from newsroom.lang_hint import detect_link_lang_hint, detect_text_lang_hint, normalise_lang_hint


class TestLangHint(unittest.TestCase):
    def test_detect_text_lang_hint_mostly_latin(self) -> None:
        text = "Prime Minister announces housing reform and detailed fiscal updates for London councils."
        self.assertEqual(detect_text_lang_hint(text), "en")

    def test_detect_text_lang_hint_mostly_cjk(self) -> None:
        text = "香港立法會通過新預算案，重點增加公共醫療資源同社區支援服務。"
        self.assertEqual(detect_text_lang_hint(text), "zh")

    def test_detect_text_lang_hint_mixed(self) -> None:
        text = "港股反彈 after Fed comments, 科技股領漲但成交未見明顯放大。"
        self.assertEqual(detect_text_lang_hint(text), "mixed")

    def test_detect_link_lang_hint_uses_existing_hint(self) -> None:
        self.assertEqual(
            detect_link_lang_hint(
                title="English title",
                description="English description",
                existing_hint="zh",
            ),
            "zh",
        )

    def test_detect_link_lang_hint_from_title_and_description(self) -> None:
        self.assertEqual(
            detect_link_lang_hint(
                title="美股急跌",
                description="Nasdaq drops after inflation surprise",
            ),
            "en",
        )

    def test_normalise_lang_hint(self) -> None:
        self.assertEqual(normalise_lang_hint("EN"), "en")
        self.assertEqual(normalise_lang_hint(" zh "), "zh")
        self.assertEqual(normalise_lang_hint("mixed"), "mixed")
        self.assertIsNone(normalise_lang_hint("fr"))


if __name__ == "__main__":
    unittest.main()
