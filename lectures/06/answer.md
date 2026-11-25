# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：glossary_search_glossary
- STEP 2 の [tool call]：
- STEP 3 の [tool call]：
- STEP 4 の [tool call]：
- 気づき・疑問：STEP 1（`uv run python -m app.agent --list-tools` と `uv run python -m app.mcp_agent --list-tools`）で、MCP 経由のツール名にサーバー名の接頭辞 `glossary_` が付くこと、`app/mcp_servers.json` に `fetch`（`uvx mcp-server-fetch==2026.11.11 --no-readability-js`）と `aws`（`https://knowledge-mcp.global.api.aws`）を追加した状態で `--list-tools` が通り、`glossary_search_glossary`・`fetch_fetch`・`aws_aws___*` がモデルに見えることを確認した。作業環境に `GEMINI_API_KEY` が無いため、実モデルで呼び出す STEP 2〜4 の `--trace` 実行は未実施。

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：基礎・実装・発展（基礎・実装はコード実装・`--list-tools`・`pytest`／MCP 配線テストまで実施し、実 LLM の `--trace` は `GEMINI_API_KEY` 不在のため未実行。発展は公式ドキュメント調査と設定比較を完了）
- やったこと：
  - 基礎：`app/mcp_server.py` の `TODO` 直下に `server.tool()(tools.get_lecture_summary)` を追加し、`app/mcp_servers.json` と `app/mcp_agent.py` を変更せずに公開ツールが増えることを確認した。
  - 実装：第3回で作成した `make_quiz` を `app/tools.py` に追加し、`app/mcp_server.py` で `server.tool()(tools.make_quiz)` として公開した。指定回が用語集に無いとき（例：`lecture=99`）は例外にせず、`第99回の用語は用語集にありません。出題できる回: 第1回, 第2回, 第3回, 第4回, 第6回` のように理由と有効な回を `return` で返すようにした。あわせて `tests/test_tools.py` と `tests/test_mcp_server.py` にテストを追加した。
  - 発展：普段利用している Claude Code を選び、公式ドキュメント（https://code.claude.com/docs/en/mcp 、調査日: 2026-11-25）で MCP サーバー設定の4項目を調べ、演習の `app/mcp_servers.json` と比較した。
- 結果：
  - ツール一覧確認：`uv run python -m app.mcp_agent --list-tools` で、`glossary_search_glossary` に加えて `glossary_get_lecture_summary` と `glossary_make_quiz` が増え、`fetch_fetch`・`aws_aws___*` と合わせて計 9 ツールがモデルに公開されることを確認した。
  - テスト実行：`uv run pytest -q` で 26 件すべて成功した。`TestModel` および `FunctionModel` を使い、MCP（stdio）経由で `glossary_get_lecture_summary(2)`・`glossary_make_quiz(2)`・`glossary_make_quiz(99)` が呼ばれ、いずれも `ToolReturnPart` として返ることを確認した。
  - `return` と `raise` の比較（`FunctionModel` + `MCPToolset` による実サーバー挙動）：
    - `return` の場合（`glossary_make_quiz(99)`）：`[tool return] glossary_make_quiz -> 第99回の用語は用語集にありません。出題できる回: 第1回, 第2回, 第3回, 第4回, 第6回` がそのままモデルに返り、有効な回の情報まで伝わる。
    - `raise ValueError(...)` の場合（仮の MCP ツールで比較）：MCP サーバー（`MCPServer`）が通常例外を `UnexpectedToolError` として捕捉し例外詳細をマスクするため、トレース上は `[tool error] ... -> Error executing tool ...` となり、どの回なら有効かという情報がモデルに届かない。
    - 実モデル（`GEMINI_API_KEY`）を使う `--trace` 実行（`"第2回では何を学びますか？"` と `"第99回のクイズを作って"`）は未実行。
  - 発展：Claude Code と `app/mcp_servers.json` の比較（調査日: 2026-11-25、出典: https://code.claude.com/docs/en/mcp ）
    - 設定を書く場所：`app/mcp_servers.json` はリポジトリ内のこの1ファイルを手動編集する。Claude Code は `claude mcp add` / `claude mcp add-json` コマンドでの登録に対応し、ファイルはプロジェクト共有がプロジェクト直下の `.mcp.json`、個人用（local / user）が `~/.claude.json` に分かれる。`"mcpServers"` をルートキーに使う点は共通。
    - コマンドで起動する書き方：`app/mcp_servers.json` は `"command"` と `"args"`（必要なら `"env"`）を書き、`"type"` は書かない。Claude Code の JSON でも `"command"` / `"args"` / `"env"` の形は同じで `"type": "stdio"` は省略できる（CLI では `claude mcp add --transport stdio <name> -- <command> [args...]` と `--` 以降に起動コマンドを渡す）。
    - URL で接続する書き方：`app/mcp_servers.json` は `"url"` を書くだけで Streamable HTTP として接続する。Claude Code の CLI は `claude mcp add --transport http <name> <url>` だが、JSON 設定では `"url"` に加えて `"type": "http"`（または `"streamable-http"` / `"sse"` / `"ws"`）が必須で、`"type"` を省略すると stdio サーバーとみなされてエラーになりスキップされる。認証用の `"headers"` や動的生成の `headersHelper` も指定できる。
    - 有効になる範囲：`app/mcp_servers.json` はこのリポジトリの `app.mcp_agent` 実行時のみ有効。Claude Code は `local`（既定：現在のプロジェクトのみ・自分専用）、`project`（現在のプロジェクトのみ・`.mcp.json` でチーム共有）、`user`（全プロジェクト・自分専用）の3スコープがあり、同名サーバーは `managed > local > project > user` の優先順位で接続される。`.mcp.json` のサーバーは対話起動時に初回承認が求められる点も異なる。
- 気づき・疑問：
  - エージェントは起動ごとに MCP サーバーへツール一覧を問い合わせるため、基礎では `app/mcp_server.py` への 1 行追加だけでエージェント無変更のまま公開ツールが増えた。
  - 存在しない回のような想定内の入力外れは、例外を投げるより案内文を `return` で返すほうが MCP サーバーの例外マスクに隠されず、モデルへ次の候補まで渡せる。
  - 同じ `"mcpServers"` 形式でもクライアントにより仕様差があり、特に Claude Code の JSON で `"url"` に `"type": "http"` を付け忘れるとエラーになる点は、設定を移植するときに気をつけたい。
