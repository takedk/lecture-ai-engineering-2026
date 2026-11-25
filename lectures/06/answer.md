# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：glossary_search_glossary
- STEP 2 の [tool call]：
- STEP 3 の [tool call]：
- STEP 4 の [tool call]：
- 気づき・疑問：STEP 1（`uv run python -m app.agent --list-tools` と `uv run python -m app.mcp_agent --list-tools`）で、MCP 経由のツール名にサーバー名の接頭辞 `glossary_` が付くこと、また `app/mcp_servers.json` に `fetch`（`uvx mcp-server-fetch==2026.11.11 --no-readability-js`）と `aws`（`https://knowledge-mcp.global.api.aws`）を追加した状態で `--list-tools` を実行し、`glossary_search_glossary`・`fetch_fetch`・`aws_aws___*` の計 7 ツールがモデルに渡る構成になることを確認した。一方、作業環境に `GEMINI_API_KEY` が無いため、実モデルを使う STEP 2〜4 の `--trace` 実行は未実施。

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：基礎 / 実装 / 発展
- やったこと：
- 結果：
- 気づき・疑問：
