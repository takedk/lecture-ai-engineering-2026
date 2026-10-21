"""ask.py のテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from app.ask import Conversation, format_usage


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(list(contents))
        usage = SimpleNamespace(
            prompt_token_count=10 * len(contents),
            candidates_token_count=5,
            thoughts_token_count=None,
            total_token_count=10 * len(contents) + 5,
        )
        return SimpleNamespace(text=f"answer{len(self.calls)}", usage_metadata=usage)


def test_conversation_sends_previous_turns():
    client = SimpleNamespace(models=FakeModels())
    conv = Conversation(client, model="fake")

    assert conv.send("LoRAとは？") == "answer1"
    assert conv.send("それをもっと短く") == "answer2"

    first, second = client.models.calls
    assert len(first) == 1
    # 2回目は「1回目の質問・1回目の答え・2回目の質問」を送る
    assert [c.role for c in second] == ["user", "model", "user"]
    assert second[0].parts[0].text == "LoRAとは？"
    assert second[1].parts[0].text == "answer1"
    assert len(conv.history) == 4


def test_reset_clears_history():
    client = SimpleNamespace(models=FakeModels())
    conv = Conversation(client, model="fake")
    conv.send("こんにちは")
    conv.reset()
    conv.send("はじめまして")
    assert len(client.models.calls[-1]) == 1


def test_format_usage():
    usage = SimpleNamespace(
        prompt_token_count=12, candidates_token_count=34, thoughts_token_count=None, total_token_count=46
    )
    assert format_usage(usage) == "[tokens] 入力 12 / 出力 34 / 合計 46"
    assert "取得できません" in format_usage(None)
