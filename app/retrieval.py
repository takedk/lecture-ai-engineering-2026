"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def expand_with_adjacent_pages(hits: list[dict], pages: list[dict] | None = None) -> list[dict]:
    """検索結果に、同じ資料内の前後1ページの本文を重複なく付ける。

    別資料のページは付けない。元のヒットは順を保ち、各ヒットの直後に前ページ・次ページを置く。
    """
    catalog = pages if pages is not None else load_pages(RAG_SOURCE)
    by_key = {(page["document_id"], page["page"]): page for page in catalog}
    seen: set[str] = set()
    results: list[dict] = []
    for hit in hits:
        pid = hit["page_id"]
        if pid not in seen:
            seen.add(pid)
            results.append({**hit, "source": hit.get("source", "search")})
        for delta, source in ((-1, "previous"), (1, "next")):
            neighbor = by_key.get((hit["document_id"], hit["page"] + delta))
            if neighbor is None or neighbor["page_id"] in seen:
                continue
            seen.add(neighbor["page_id"])
            results.append({**neighbor, "source": source, "anchor_page_id": pid})
    return results


_QUERY_STOPWORDS = {
    "説明", "場合", "以上", "以下", "今回", "内容", "確認", "質問", "講座", "講義",
    "資料", "本文", "指定", "対象", "全部", "もの", "こと", "ため", "よう", "とき",
}


def query_terms(text: str) -> set[str]:
    """質問や本文から、照合用の用語（英数字・カタカナ・漢字2文字以上）を取り出す。"""
    terms = set()
    for token in re.findall("[A-Za-z0-9_]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+", text):
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            token = token.casefold()
        if len(token) >= 2 and token not in _QUERY_STOPWORDS:
            terms.add(token)
    return terms


def overlapping_terms(query: str, page: dict) -> list[str]:
    """質問と、ページのタイトル・本文で重なる用語を返す。"""
    page_text = f"{page.get('title', '')}\n{page.get('content', '')}"
    return sorted(query_terms(query) & query_terms(page_text))


def search(query: str, *, include_adjacent: bool = False) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。

    include_adjacent=True のときは、同じ資料の前後1ページも付ける。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        hits = [dict(row) for row in rows]
    if include_adjacent:
        return expand_with_adjacent_pages(hits)
    return hits


def search_hybrid(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと、質問語と重なるグラフ先だけを返す。"""
    graph = search_graph(query)
    seeds = graph["seeds"]
    adjacent = [page for page in expand_with_adjacent_pages(seeds) if page.get("source") != "search"]
    seen = {seed["page_id"] for seed in seeds} | {page["page_id"] for page in adjacent}
    expanded = []
    dropped = 0
    for page in graph["expanded"]:
        if page["page_id"] in seen:
            continue
        overlap = overlapping_terms(query, page)
        if not overlap:
            dropped += 1
            continue
        seen.add(page["page_id"])
        expanded.append({**page, "source": "graph", "overlap_terms": overlap})
    return {
        "seeds": seeds,
        "adjacent": adjacent,
        "expanded": expanded,
        "max_hops": graph["max_hops"],
        "dropped_without_query_terms": dropped,
    }


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


def _print_pages(results: list[dict], heading: str) -> None:
    for i, result in enumerate(results, 1):
        source = result.get("source", "search")
        print(f"\n{'=' * 60}\n{heading} {i} ({source})\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        if "overlap_terms" in result:
            print(f"重なった用語: {', '.join(result['overlap_terms'])}")
        print(f"\n{result['content']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--adjacent", action="store_true", help="同じ資料の前後1ページも付ける")
    mode.add_argument("--hybrid", action="store_true", help="前後ページと、質問語と重なるグラフ先も付ける")
    args = parser.parse_args()
    if args.hybrid:
        result = search_hybrid(args.query)
        _print_pages(result["seeds"], "入口")
        _print_pages(result["adjacent"], "前後ページ")
        _print_pages(result["expanded"], "グラフ")
        print(f"\n質問語と重ならず除外したグラフ先: {result['dropped_without_query_terms']} 件")
    else:
        _print_pages(search(args.query, include_adjacent=args.adjacent), "検索結果")
