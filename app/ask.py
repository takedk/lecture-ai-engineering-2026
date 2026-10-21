"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --temperature 1.5 "LoRAとは何ですか？"
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（宿題・実装）

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# 会話モードを終わらせる入力
EXIT_WORDS = {"exit", "quit", "q", "終了"}


def format_usage(usage) -> str:
    """usage_metadata を「入力 / 出力 トークン数」の1行にする。"""
    if usage is None:
        return "[tokens] 取得できませんでした"
    prompt = usage.prompt_token_count or 0
    output = usage.candidates_token_count or 0
    line = f"[tokens] 入力 {prompt} / 出力 {output}"
    # 思考するモデルでは、出力とは別に思考トークンが数えられる
    thoughts = getattr(usage, "thoughts_token_count", None)
    if thoughts:
        line += f"（思考 {thoughts}）"
    total = usage.total_token_count
    if total is not None:
        line += f" / 合計 {total}"
    return line


def make_config(temperature: float | None) -> types.GenerateContentConfig | None:
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL, contents=question, config=make_config(temperature)
    )

    # 第1回 演習: response.usage_metadata から入力・出力トークン数を表示する
    print(format_usage(response.usage_metadata))

    return response.text


class Conversation:
    """会話履歴を保ったまま続けて質問する（第1回 宿題・実装）。

    LLM の API は状態を持たないので、毎回これまでのやりとりを全部 contents として送る。
    そのため、ターンが進むほど入力トークン数が増える。
    """

    def __init__(self, client, model: str = MODEL, temperature: float | None = None):
        self.client = client
        self.model = model
        self.config = make_config(temperature)
        self.history: list[types.Content] = []

    def send(self, question: str) -> str:
        user_turn = types.Content(role="user", parts=[types.Part(text=question)])
        response = self.client.models.generate_content(
            model=self.model, contents=[*self.history, user_turn], config=self.config
        )
        # 失敗したターンは履歴に残さない（成功したときだけ追加する）
        answer = response.text or ""
        self.history.append(user_turn)
        self.history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
        self.last_usage = response.usage_metadata
        return answer

    def reset(self) -> None:
        self.history.clear()


def chat(temperature: float | None = None) -> None:
    conv = Conversation(genai.Client(api_key=get_api_key()), temperature=temperature)
    print("会話モードです。exit で終了、/reset で履歴を消します。\n")
    while True:
        try:
            question = input("あなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        if question == "/reset":
            conv.reset()
            print("履歴を消しました。\n")
            continue
        try:
            answer = conv.send(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message) + "\n")
            continue
        print(f"\nモデル> {answer}")
        print(format_usage(conv.last_usage) + f" / 履歴 {len(conv.history) // 2} ターン\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--temperature", type=float, default=None, help="生成の温度（省略時はモデルの既定値）")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.temperature is not None:
        print(f"[temperature] {args.temperature}")
    if args.chat:
        chat(args.temperature)
        return
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
