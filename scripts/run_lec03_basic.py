"""第3回 宿題（基礎）: INSTRUCTIONS とツールの docstring を1か所ずつ変えて、同じ質問への挙動を比べる。

app/ のファイルは書き換えず、変更版のエージェントをこのスクリプトの中で組み立てる。
変えるのは毎回1か所だけ。各条件を REPEAT 回ずつ実行し、ツール呼び出し・requests・回答を
lectures/03/results/basic/ に保存する。

    uv run python scripts/run_lec03_basic.py
"""

import functools
import json
import sys
import time
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ToolCallPart

from app import tools
from app.agent import INSTRUCTIONS, build_model

OUT = Path(__file__).resolve().parent.parent / "lectures" / "03" / "results" / "basic"
REPEAT = 3

LINE_SEARCH = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
LINE_LECTURE = "- 回答の最後に、関係する回（第N回）を示す。\n"
DOC_ONE = 'query には調べたい用語を1つだけ入れる（例: "LoRA"）。'
DOC_MANY = 'query には調べたい用語をまとめて入れる（例: "LoRA QLoRA"）。'
DOC_BEFORE = "AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について\n答える前に使うこと。"


def with_doc(func, old: str, new: str):
    """docstring の一部だけを置き換えたツールを作る（関数名・引数は元のまま）。"""
    assert old in func.__doc__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__doc__ = func.__doc__.replace(old, new)
    return wrapper


def build(instructions: str = INSTRUCTIONS, search=tools.search_glossary) -> Agent:
    a = Agent(instructions=instructions)
    a.tool_plain(search)
    a.tool_plain(tools.get_lecture_summary)
    a.tool_plain(tools.make_quiz)
    return a


assert LINE_SEARCH in INSTRUCTIONS and LINE_LECTURE in INSTRUCTIONS

CONDITIONS = {
    # (条件名, エージェント, 質問)
    "A_baseline_lora": (build(), "LoRAとQLoRAの違いは？"),
    "A_no_search_line_lora": (build(INSTRUCTIONS.replace(LINE_SEARCH, "")), "LoRAとQLoRAの違いは？"),
    "A_baseline_lecstart": (build(), "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"),
    "A_no_search_line_lecstart": (
        build(INSTRUCTIONS.replace(LINE_SEARCH, "")),
        "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？",
    ),
    "B_doc_many_lora": (build(search=with_doc(tools.search_glossary, DOC_ONE, DOC_MANY)), "LoRAとQLoRAの違いは？"),
    "B_baseline_3terms": (build(), "LoRA、QLoRA、PEFTの関係を教えて"),
    "B_doc_many_3terms": (
        build(search=with_doc(tools.search_glossary, DOC_ONE, DOC_MANY)),
        "LoRA、QLoRA、PEFTの関係を教えて",
    ),
    # A の追加: INSTRUCTIONS の行に加えて、docstring の「答える前に使うこと」も消す
    "A2_no_line_no_doc_lora": (
        build(INSTRUCTIONS.replace(LINE_SEARCH, ""), with_doc(tools.search_glossary, DOC_BEFORE, "")),
        "LoRAとQLoRAの違いは？",
    ),
    "A2_no_line_no_doc_lecstart": (
        build(INSTRUCTIONS.replace(LINE_SEARCH, ""), with_doc(tools.search_glossary, DOC_BEFORE, "")),
        "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？",
    ),
    "C_quiz_at_end_lora": (
        build(INSTRUCTIONS.replace(LINE_LECTURE, "- 回答の最後に、内容の確認問題を1問出す。\n")),
        "LoRAとQLoRAの違いは？",
    ),
}


def run_once(agent: Agent, question: str):
    for wait in (0, 20, 40, 60, 90):
        time.sleep(wait)
        try:
            return agent.run_sync(question, model=build_model())
        except ModelHTTPError as e:
            if e.status_code not in (429, 503):
                raise
            print(f"  retry after HTTP {e.status_code}")
        except ModelAPIError as e:  # 接続タイムアウトなど
            print(f"  retry after {e}")
    raise RuntimeError("API が混み合っていて実行できませんでした")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 引数で条件名を渡すと、その条件だけを実行して summary.json に追記する
    selected = sys.argv[1:] or list(CONDITIONS)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in selected:
        agent, question = CONDITIONS[name]
        runs = []
        for i in range(REPEAT):
            print(f"{name} #{i + 1}")
            result = run_once(agent, question)
            calls = [
                {"tool": p.tool_name, "args": p.args_as_dict()}
                for m in result.all_messages()
                for p in m.parts
                if isinstance(p, ToolCallPart)
            ]
            runs.append(
                {
                    "tool_calls": calls,
                    "requests": result.usage.requests,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "output": result.output,
                }
            )
            time.sleep(4)  # レート制限よけ
        summary[name] = {"question": question, "runs": runs}
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / f"{name}.json").write_text(json.dumps(summary[name], ensure_ascii=False, indent=2), encoding="utf-8")
    for name in CONDITIONS:
        if name not in summary:
            continue
        s = summary[name]
        print(f"\n== {name}: {s['question']}")
        for r in s["runs"]:
            args = ", ".join(f"{c['tool']}({c['args']})" for c in r["tool_calls"]) or "(ツール呼び出しなし)"
            print(f"  requests={r['requests']} in={r['input_tokens']} out={r['output_tokens']} | {args}")


if __name__ == "__main__":
    main()
