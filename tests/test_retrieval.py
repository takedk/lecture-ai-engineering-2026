"""前後ページの付与と、質問語によるグラフ先の絞り込み。APIは呼ばない。"""

from app.retrieval import (
    expand_with_adjacent_pages,
    load_pages,
    overlapping_terms,
    query_terms,
    search,
    search_hybrid,
)


def test_query_terms_keeps_content_words_and_drops_stopwords():
    terms = query_terms("第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？")
    assert {"lora", "qlora", "固定", "学習"} <= terms
    assert "説明" not in terms


def test_expand_adds_prev_next_in_the_same_document():
    pages = {page["page_id"]: page for page in load_pages()}
    results = expand_with_adjacent_pages([pages["lec02-main-p052"]])
    assert [page["page_id"] for page in results] == [
        "lec02-main-p052",
        "lec02-main-p051",
        "lec02-main-p053",
    ]
    assert [page["source"] for page in results] == ["search", "previous", "next"]
    assert results[1]["content"]
    assert "本体は固定し" in results[1]["content"]
    assert results[1]["lecture"] == 2
    assert results[1]["page"] == 51
    assert results[2]["page"] == 53
    assert results[1]["anchor_page_id"] == "lec02-main-p052"


def test_expand_skips_duplicates_and_does_not_cross_documents():
    pages = load_pages()
    by_id = {page["page_id"]: page for page in pages}
    first_lec03 = min(
        (page for page in pages if page["document_id"] == "lec03-main"),
        key=lambda page: page["page"],
    )
    last_lec02 = max(
        (page for page in pages if page["document_id"] == "lec02-main"),
        key=lambda page: page["page"],
    )
    assert first_lec03["page"] == 1
    # 第2回の最終ページの次に、第3回の1ページ目を付けない
    only_last = expand_with_adjacent_pages([last_lec02])
    assert first_lec03["page_id"] not in {page["page_id"] for page in only_last}
    assert all(page["document_id"] == "lec02-main" for page in only_last)

    # 隣同士が両方ヒットしても、同じページは一度だけ返す
    results = expand_with_adjacent_pages(
        [by_id["lec02-main-p052"], by_id["lec02-main-p053"]]
    )
    ids = [page["page_id"] for page in results]
    assert ids == [
        "lec02-main-p052",
        "lec02-main-p051",
        "lec02-main-p053",
        "lec02-main-p054",
    ]
    assert len(ids) == len(set(ids))


def test_expand_handles_first_and_last_pages():
    pages = [page for page in load_pages() if page["document_id"] == "lec02-main"]
    pages.sort(key=lambda page: page["page"])
    first, last = pages[0], pages[-1]
    assert [page["source"] for page in expand_with_adjacent_pages([first])] == ["search", "next"]
    assert [page["source"] for page in expand_with_adjacent_pages([last])] == ["search", "previous"]


def test_adjacent_helps_memory_question_when_seed_is_page_39():
    pages = {page["page_id"]: page for page in load_pages()}
    results = expand_with_adjacent_pages([pages["lec03-main-p039"]])
    assert [page["page_id"] for page in results] == [
        "lec03-main-p039",
        "lec03-main-p038",
        "lec03-main-p040",
    ]
    next_page = results[2]
    assert "全部は残さない" in next_page["content"]
    assert "長期メ" in next_page["content"] or "長期メモリ" in next_page["content"]


def test_overlapping_terms_on_real_pages():
    pages = {page["page_id"]: page for page in load_pages()}
    lora_q = "LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？"
    assert {"lora", "qlora", "固定", "学習"} <= set(overlapping_terms(lora_q, pages["lec02-main-p051"]))
    assert overlapping_terms(lora_q, pages["lec02-main-p053"]) == []

    memory_q = "コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか"
    assert "コンパクション" in overlapping_terms(memory_q, pages["lec03-main-p039"])
    assert "セッション" in overlapping_terms(memory_q, pages["lec03-main-p039"])
    assert {"メモ", "メモリ", "長期"} <= set(overlapping_terms(memory_q, pages["lec03-main-p040"]))


def test_search_include_adjacent_uses_mocked_hits(monkeypatch):
    pages = {page["page_id"]: page for page in load_pages()}
    hits = [pages["lec02-main-p052"]]

    def fake_search_db(query, *, include_adjacent=False):
        raise AssertionError("search本体を直接試す")

    monkeypatch.setattr("app.retrieval.embed_text", lambda *args, **kwargs: [0.0])
    monkeypatch.setattr("app.retrieval.open_index", _fake_index(hits))
    results = search("LoRA QLoRA", include_adjacent=True)
    assert [page["page_id"] for page in results] == [
        "lec02-main-p052",
        "lec02-main-p051",
        "lec02-main-p053",
    ]
    bare = search("LoRA QLoRA")
    assert [page["page_id"] for page in bare] == ["lec02-main-p052"]


class _FakeRows:
    def __init__(self, hits):
        self.hits = hits

    def __iter__(self):
        return iter(self.hits)


class _FakeDb:
    def __init__(self, hits):
        self.hits = hits

    def execute(self, *_args, **_kwargs):
        return _FakeRows(self.hits)

    def close(self):
        return None


class _fake_index:
    def __init__(self, hits):
        self.hits = hits

    def __call__(self, _path):
        return self

    def __enter__(self):
        return _FakeDb(self.hits)

    def __exit__(self, *_args):
        return False


def test_search_hybrid_keeps_query_overlap_and_drops_the_rest(monkeypatch):
    pages = {page["page_id"]: page for page in load_pages()}
    seeds = [{**pages["lec03-main-p039"], "distance": 0.1}]
    related = pages["lec03-main-p013"]  # メモリの説明。質問語と重なる
    unrelated = {
        **pages["lec02-main-p054"],
        "content": "手本データの作り方だけを述べたページ",
        "title": "手本データの作り方",
    }

    def fake_graph(query):
        return {
            "seeds": [{**seeds[0], "hops": 0, "path": [{"id": seeds[0]["page_id"], "title": seeds[0]["title"]}]}],
            "expanded": [
                {**related, "hops": 2, "path": []},
                {**unrelated, "hops": 2, "path": []},
                {**pages["lec03-main-p040"], "hops": 1, "path": []},  # 前後ページでも付くのでexpandedからは除く
            ],
            "max_hops": 2,
        }

    monkeypatch.setattr("app.retrieval.search_graph", fake_graph)
    result = search_hybrid("コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか")
    assert [page["page_id"] for page in result["seeds"]] == ["lec03-main-p039"]
    assert [page["page_id"] for page in result["adjacent"]] == [
        "lec03-main-p038",
        "lec03-main-p040",
    ]
    assert [page["page_id"] for page in result["expanded"]] == ["lec03-main-p013"]
    assert "メモリ" in result["expanded"][0]["overlap_terms"]
    assert result["dropped_without_query_terms"] == 1
