"""第3回 宿題（発展 a）: 同じ質問を ADK 版と Pydantic AI 版に投げ、ツール呼び出しと回答を並べる。

adk run は対話用なので、ここでは ADK の Runner を直接使って、イベントから
function_call（ツール呼び出し）と LLM 呼び出し回数を取り出す。

    PYTHONPATH=. uv run --with google-adk python scripts/run_lec03_adk_compare.py
"""

import asyncio
import json
import time
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ToolCallPart

from adk_agent.glossary_agent.agent import root_agent
from app.agent import agent as pai_agent, build_model

OUT = Path(__file__).resolve().parent.parent / "lectures" / "03" / "results" / "adk_compare.json"
QUESTIONS = [
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "LoRAとQLoRAの違いは？",
    "第2回の内容から確認問題を1問出して",
    "第3回では何を学ぶ？",
]


async def run_adk(question: str) -> dict:
    runner = InMemoryRunner(agent=root_agent, app_name="glossary_agent")
    session = await runner.session_service.create_session(app_name="glossary_agent", user_id="u")
    calls, texts, llm_responses = [], [], 0
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        if event.author != root_agent.name or not event.content or event.content.role != "model":
            continue
        llm_responses += 1  # モデルが返したイベント＝ LLM 呼び出し1回分
        for part in event.content.parts or []:
            if part.function_call:
                calls.append({"tool": part.function_call.name, "args": dict(part.function_call.args or {})})
            elif part.text and not part.thought:
                texts.append(part.text)
    return {"tool_calls": calls, "llm_responses": llm_responses, "output": "".join(texts)}


def run_pai(question: str) -> dict:
    for wait in (0, 20, 40, 60):
        time.sleep(wait)
        try:
            result = pai_agent.run_sync(question, model=build_model())
            break
        except ModelHTTPError as e:
            if e.status_code not in (429, 503):
                raise
        except ModelAPIError:  # 接続タイムアウトなど
            pass
    else:
        raise RuntimeError("Pydantic AI: API が混み合っていて実行できませんでした")
    calls = [
        {"tool": p.tool_name, "args": p.args_as_dict()}
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    return {"tool_calls": calls, "llm_responses": result.usage.requests, "output": result.output}


async def run_adk_retry(question: str) -> dict:
    for wait in (0, 20, 40, 60):
        await asyncio.sleep(wait)
        try:
            return await run_adk(question)
        except Exception as e:  # ADK は google-genai の例外をそのまま投げる
            if not any(k in str(e) for k in ("503", "429", "UNAVAILABLE", "Timeout", "timed out")):
                raise
            print(f"  ADK retry: {str(e)[:80]}")
    raise RuntimeError("ADK: API が混み合っていて実行できませんでした")


def main() -> None:
    rows = []
    for q in QUESTIONS:
        print(f"Q: {q}")
        adk = asyncio.run(run_adk_retry(q))
        time.sleep(4)
        pai = run_pai(q)
        time.sleep(4)
        rows.append({"question": q, "adk": adk, "pydantic_ai": pai})
        for name, r in (("ADK", adk), ("PydanticAI", pai)):
            args = ", ".join(f"{c['tool']}({c['args']})" for c in r["tool_calls"]) or "(なし)"
            print(f"  {name:10s} llm={r['llm_responses']} | {args}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
