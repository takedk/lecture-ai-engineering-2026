"""第3回 宿題（発展 a）: 演習のエージェントを Google ADK で組み立て直す。

指示（INSTRUCTIONS）とツール（app/tools.py の関数）は Pydantic AI 版とまったく同じものを使い、
エージェントを組み立てる部分だけを書き直している。

    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    model=MODEL,  # 既定は gemini-3.5-flash-lite（Pydantic AI 版と同じ。AIE_MODEL で切り替え）
    name="glossary_agent",
    description="AIE2026 の用語集と各回の要点を調べて答える学習支援エージェント",
    instruction=INSTRUCTIONS,
    # 関数をそのまま渡すと、ADK が名前・docstring・型ヒントからツールの説明を作る
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
