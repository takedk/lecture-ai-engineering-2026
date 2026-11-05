"""第3回 宿題（実装）: save_learning_log と承認の流れのテスト（APIキー不要）。"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approval


def test_save_learning_log_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    msg = tools.save_learning_log("第3回 / ツール", "- 学んだこと", lecture=3)
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1 and files[0].name in msg
    text = files[0].read_text(encoding="utf-8")
    assert "# 第3回 / ツール" in text and "第3回" in text and "- 学んだこと" in text
    assert "/" not in files[0].name.split("-", 2)[-1]  # 見出しの / はファイル名に残さない


def test_save_is_paused_until_approved(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])
    result = agent.run_sync("今日の学びを保存して", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not list(tmp_path.glob("*.md"))  # 承認前には書き込まない


def test_approved_save_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])
    result = run_with_approval(agent, "今日の学びを保存して", model, answer=lambda _: "y")
    assert isinstance(result.output, str)
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_denied_save_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])
    result = run_with_approval(agent, "今日の学びを保存して", model, answer=lambda _: "n")
    assert isinstance(result.output, str)
    assert not list(tmp_path.glob("*.md"))
