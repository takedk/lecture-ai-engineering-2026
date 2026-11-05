"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
import re
from datetime import datetime
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語集から確認問題を1問作り、問題と答えを返す。

    「第2回の確認問題を出して」「復習のクイズを1問」のように、確認問題や小テストを
    求められたときに使う。lecture には対象の回の番号を入れる（例: 2）。
    返り値は JSON で、question（問題文）と answer（正解）と term（出題した用語）を持つ。
    出題するときは answer をすぐには見せず、利用者が答えてから示すこと。
    """
    candidates = [e for e in load_glossary() if e["lecture"] == lecture]
    if not candidates:
        available = sorted({e["lecture"] for e in load_glossary()})
        return (
            f"第{lecture}回の用語は用語集にありません。"
            f"出題できる回: {', '.join(f'第{n}回' for n in available)}"
        )
    entry = random.choice(candidates)
    return json.dumps(
        {
            "question": f"「{entry['term']}」の意味は？",
            "answer": entry["definition"],
            "term": entry["term"],
            "lecture": lecture,
        },
        ensure_ascii=False,
    )


LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"


def save_learning_log(title: str, content: str, lecture: int | None = None) -> str:
    """学習の記録（学んだこと・つまずいた点・次にやること）を learning_log/ フォルダに Markdown で保存する。

    利用者が「今日の学びを記録して」「メモを保存して」のように、記録や保存を
    はっきり頼んだときだけ使う。頼まれていないのに自分から保存しない。
    title には記録の短い見出し、content には保存する本文（Markdown）を入れる。
    lecture には関係する回の番号を入れる（分からなければ省略する）。
    保存はファイルを書き込む操作なので、実行前に利用者の承認が必要になる。
    """
    title = title.strip() or "学習の記録"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    # ファイル名に使えない文字を「-」に置き換え、長すぎる見出しは切る
    safe = re.sub(r'[\\/:*?"<>|\s]+', "-", title).strip("-")[:40] or "log"
    path = LOG_DIR / f"{now:%Y%m%d-%H%M%S}-{safe}.md"
    header = f"# {title}\n\n- 日時: {now:%Y-%m-%d %H:%M}\n"
    if lecture is not None:
        header += f"- 関係する回: 第{lecture}回\n"
    path.write_text(f"{header}\n{content.strip()}\n", encoding="utf-8")
    return f"保存しました: learning_log/{path.name}"
