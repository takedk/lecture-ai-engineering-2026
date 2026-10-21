"""第1回 宿題（基礎・発展）: 応答時間とトークン数を計測して CSV に保存する。

使い方:
    uv run python -m scripts.measure_lec01 --plan basic   # 基礎: 同じ質問を3回ずつ、モデル・temperature を変える
    uv run python -m scripts.measure_lec01 --plan advanced  # 発展: 複数のモデル × 複数の質問
    uv run python -m scripts.measure_lec01 --plan stream  # 発展: ストリーミングで最初のトークンまでの時間も測る
    uv run python -m scripts.measure_lec01 --summary      # 保存済みの CSV を集計して表示する

同じ (plan, モデル, 質問, temperature, 回) がすでに CSV にあれば飛ばすので、途中で止まっても再実行で続きから測れる。

無料枠のレート制限を超えないよう、モデルごとに呼び出しの間隔をあける。
エラー（429・503・通信エラー・タイムアウト）も1行として記録する（同じ回の再試行はしない）。
APIキーは環境変数 GEMINI_API_KEY から読むだけで、どこにも書き出さない。
"""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION, make_config
from app.config import get_api_key

OUT_DIR = Path(__file__).resolve().parent.parent / "lectures" / "01" / "results"
CSV_PATH = OUT_DIR / "measurements.csv"
ANSWERS_PATH = OUT_DIR / "answers.jsonl"

QUESTIONS = {
    "lora": "LoRAとは何ですか？",
    "levels": DEFAULT_QUESTION,
    "short": "日本の首都はどこですか？一言で答えてください。",
}

# 無料枠の上限（講座の実測）から決めた呼び出し間隔（秒）。毎分の上限より少し余裕を持たせる
INTERVAL = {
    "gemini-3.5-flash-lite": 5,  # 毎分15
    "gemini-3.1-flash-lite": 5,
    "gemini-3.8-flash": 15,  # 毎分5、1日10〜20前後
}

TIMEOUT_MS = 90_000  # 1回の呼び出しの上限。サーバーが応答しないまま数分止まることがあったため

# plan -> [(model, question_id, temperature, 回数)]。stream の plan はストリーミングで呼ぶ
STREAM_PLANS = {"stream"}
PLANS = {
    "basic": [
        ("gemini-3.5-flash-lite", "levels", None, 3),
        ("gemini-3.5-flash-lite", "levels", 0.0, 3),
        ("gemini-3.5-flash-lite", "levels", 2.0, 3),
        ("gemini-3.1-flash-lite", "levels", None, 3),
        ("gemini-3.8-flash", "levels", None, 3),
    ],
    "advanced": [
        ("gemini-3.5-flash-lite", "lora", None, 3),
        ("gemini-3.5-flash-lite", "short", None, 3),
        ("gemini-3.1-flash-lite", "lora", None, 3),
        ("gemini-3.1-flash-lite", "short", None, 3),
        ("gemini-3.8-flash", "lora", None, 1),
        ("gemini-3.8-flash", "short", None, 1),
    ],
    "stream": [
        ("gemini-3.5-flash-lite", "lora", None, 3),
        ("gemini-3.5-flash-lite", "short", None, 3),
        ("gemini-3.1-flash-lite", "lora", None, 3),
        ("gemini-3.1-flash-lite", "short", None, 3),
    ],
}

FIELDS = [
    "time_utc", "plan", "mode", "model", "question_id", "temperature", "run", "status",
    "latency_s", "ttft_s", "input_tokens", "output_tokens", "thought_tokens", "total_tokens",
    "answer_chars", "error",
]


def measure_once(client, model: str, question: str, temperature: float | None, stream: bool) -> dict:
    start = time.perf_counter()
    ttft = None
    try:
        if stream:
            chunks = []
            usage = None
            for chunk in client.models.generate_content_stream(
                model=model, contents=question, config=make_config(temperature)
            ):
                if ttft is None and chunk.text:
                    ttft = time.perf_counter() - start
                chunks.append(chunk.text or "")
                usage = chunk.usage_metadata or usage
            text = "".join(chunks)
        else:
            response = client.models.generate_content(
                model=model, contents=question, config=make_config(temperature)
            )
            text = response.text or ""
            usage = response.usage_metadata
    except errors.APIError as e:
        return {
            "status": e.code, "latency_s": round(time.perf_counter() - start, 2),
            "error": (e.message or "")[:120], "answer": "",
        }
    except Exception as e:  # 通信エラーやタイムアウト（httpx など）
        return {
            "status": "network", "latency_s": round(time.perf_counter() - start, 2),
            "error": f"{type(e).__name__}: {e}"[:120], "answer": "",
        }
    latency = time.perf_counter() - start
    return {
        "status": 200, "latency_s": round(latency, 2),
        "ttft_s": "" if ttft is None else round(ttft, 2),
        "input_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count,
        "thought_tokens": usage.thoughts_token_count or 0, "total_tokens": usage.total_token_count,
        "answer_chars": len(text), "error": "", "answer": text,
    }


def done_keys() -> set[tuple]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open() as f:
        return {(r["plan"], r["model"], r["question_id"], r["temperature"], r["run"]) for r in csv.DictReader(f)}


def run_plan(plan: str) -> None:
    client = genai.Client(api_key=get_api_key(), http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    stream = plan in STREAM_PLANS
    done = done_keys()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f, ANSWERS_PATH.open("a") as af:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for model, qid, temp, times in PLANS[plan]:
            for i in range(1, times + 1):
                if (plan, model, qid, "" if temp is None else str(temp), str(i)) in done:
                    continue
                row = measure_once(client, model, QUESTIONS[qid], temp, stream)
                row.update(
                    time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    plan=plan, mode="stream" if stream else "normal", model=model, question_id=qid,
                    temperature="" if temp is None else temp, run=i,
                )
                writer.writerow(row)
                f.flush()
                af.write(json.dumps({k: row[k] for k in ("plan", "mode", "model", "question_id", "temperature", "run", "status", "answer")}, ensure_ascii=False) + "\n")
                af.flush()
                print(f"{model:24} {qid:7} temp={row['temperature']!s:4} #{i} status={row['status']} "
                      f"{row['latency_s']}s ttft={row.get('ttft_s', '')} in={row.get('input_tokens')} out={row.get('output_tokens')} "
                      f"think={row.get('thought_tokens')} {row.get('error', '')}", flush=True)
                time.sleep(INTERVAL.get(model, 5))


def summary() -> str:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["mode"], r["model"], r["question_id"], r["temperature"]), []).append(r)
    lines = [
        "| 呼び方 | モデル | 質問 | temperature | 成功/試行 | 平均応答時間(秒) | 最小〜最大(秒) | 平均TTFT(秒) | 平均入力 | 平均出力 | 平均思考 | 出力トークン/秒 | 失敗 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (mode, model, qid, temp), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "200"]
        failed = [r["status"] for r in rs if r["status"] != "200"]
        fail_txt = "、".join(f"{c}×{failed.count(c)}" for c in sorted(set(failed))) or "なし"
        head = f"| {mode} | {model} | {qid} | {temp or '既定'} | {len(ok)}/{len(rs)} |"
        if not ok:
            lines.append(head + " - | - | - | - | - | - | - | " + fail_txt + " |")
            continue
        lat = [float(r["latency_s"]) for r in ok]
        ttft = [float(r["ttft_s"]) for r in ok if r["ttft_s"]]
        inp = [int(r["input_tokens"]) for r in ok]
        out = [int(r["output_tokens"]) for r in ok]
        think = [int(r["thought_tokens"]) for r in ok]
        ttft_txt = f"{statistics.mean(ttft):.1f}" if ttft else "-"
        lines.append(
            head + f" {statistics.mean(lat):.1f} | {min(lat):.1f}〜{max(lat):.1f} | {ttft_txt} |"
            f" {statistics.mean(inp):.0f} | {statistics.mean(out):.0f} | {statistics.mean(think):.0f} |"
            f" {sum(out) / sum(lat):.0f} | {fail_txt} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=PLANS)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.plan:
        run_plan(args.plan)
    if args.summary:
        print(summary())


if __name__ == "__main__":
    main()
