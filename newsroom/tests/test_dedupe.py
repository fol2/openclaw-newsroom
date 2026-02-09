from newsroom.dedupe import semantic_match, title_features


def test_semantic_dedupe_catches_cjk_duplicates_with_single_strong_anchor() -> None:
    a = "史雲頓違規派人參賽被逐出 Vertu Trophy 盧頓獲判「翻盤」晉級八強鬥普利茅夫"
    b = "史雲頓違規派人上陣遭逐出英錦標 盧頓獲判晉級八強"
    m = semantic_match(title_features(a), title_features(b))
    assert m.is_duplicate


def test_semantic_dedupe_catches_cjk_duplicates_even_when_wording_differs() -> None:
    a = "【進軍債市】港交所擬擴展固定收益及大宗商品業務 應對全球投資者轉向"
    b = "港交所擬擴展債券及大宗商品業務 吸引全球投資者分散美資風險"
    m = semantic_match(title_features(a), title_features(b))
    assert m.is_duplicate


def test_semantic_dedupe_does_not_overmatch_unrelated_titles() -> None:
    a = "【加沙重聚】拉法口岸重開：民眾闊別親朋喜極而泣 惟空襲未停戰雲籠罩"
    b = "金價強勢衝破五千美元關口；亞太股市逆勢走高，同美股科技股拋售潮「分道揚鑣」"
    m = semantic_match(title_features(a), title_features(b))
    assert not m.is_duplicate

