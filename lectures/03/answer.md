# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  - 「AIE2026の講座の宿題にある「3レベル」とは何ですか？」を両方に投げた（`lectures/03/results/ex2_*.txt`）。
  - 素のLLM（`app.ask`）は「手元に教材がないので確言できない」と前置きしたうえで、「レベル1 基礎・入門／レベル2 応用・実践／レベル3 発展・プロ級」のような一般論を2パターン並べ、最後は「指示書を見直すか講師に聞いて」で終わった。講座の中身は答えられていない。
  - エージェント（`app.agent`）は用語集の `homework-levels` を引いて「基礎：講義内容の確認／実装：自分でコードを書く／発展：調査や試行」「どれか1つで完了」「原則として採点しない」と、講座の定義どおりに答え、最後に「関係する回：第1回」を付けた。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - 「LoRAとQLoRAの違いは？」→ `search_glossary({'query': 'LoRA'})` と `search_glossary({'query': 'QLoRA'})` の2回。docstring の「用語を1つだけ入れる」どおり、用語ごとに分けて呼んでいた。
- 1つの質問での LLM 呼び出し回数（`requests`）：
  - 2回（`requests=2, tool_calls=2`）。1回目の応答で2つの検索をまとめて依頼し、結果を受け取った2回目で回答を書いている。検索が2回あっても LLM 呼び出しは2回で済んでいた。
- 追加したツールと、docstring に書いた説明：
  - `make_quiz(lecture: int) -> str`：用語集から指定した回の用語を `random.choice` で1つ選び、`{"question": "「PEFT」の意味は？", "answer": 定義, "term", "lecture"}` を JSON で返す。用語がない回（第0回など）は「第0回の用語は用語集にありません。出題できる回: 第1回, 第2回, 第3回, 第5回」と返す。
  - docstring：「指定した回の用語集から確認問題を1問作り、問題と答えを返す。『第2回の確認問題を出して』『復習のクイズを1問』のように、確認問題や小テストを求められたときに使う。lecture には対象の回の番号を入れる（例: 2）。…出題するときは answer をすぐには見せず、利用者が答えてから示すこと。」
  - 結果：「第2回の内容から確認問題を1問出して」で `make_quiz({'lecture': 2})` が呼ばれ、答えは伏せたまま「「PEFT」の意味は何でしょうか？」と出題した（`requests=2`）。INSTRUCTIONS は変えずに、docstring だけで呼ばれた。`uv run pytest -q` は通過。

## 宿題

- 取り組んだレベル：基礎・実装・発展（a ADK）の3つすべて

### 基礎：INSTRUCTIONS と docstring を1か所ずつ変えて比べる

- やったこと：
  - `scripts/run_lec03_basic.py` で、変更版のエージェントをスクリプトの中で組み立てた（`app/` は書き換えない）。変えるのは毎回1か所だけにし、各条件を3回ずつ実行して、ツール呼び出し・`requests`・入力トークン・回答を `lectures/03/results/basic/summary.json` に保存した。
  - モデルは `gemini-3.5-flash-lite`（既定）。
- 結果（3回とも同じ傾向だったので代表値。入力トークンは3回の値がほぼ同じ）：

| 条件 | 質問 | ツール呼び出し | requests | 入力トークン |
|---|---|---|---|---|
| 元のまま | LoRAとQLoRAの違いは？ | search_glossary×2（LoRA / QLoRA） | 2 | 1514 |
| A: INSTRUCTIONS の「必ず search_glossary で確認する」行を削除 | 同上 | search_glossary×2（変化なし） | 2 | 1466 |
| A2: A に加えて docstring の「答える前に使うこと」も削除 | 同上 | search_glossary×2（変化なし） | 2 | 1406 |
| 元のまま | lecNN-start タグとは？ | search_glossary×1 | 2 | 1101 |
| A: 行を削除 | 同上 | search_glossary×1（変化なし） | 2 | 1053 |
| A2: 行と docstring の一文を削除 | 同上 | search_glossary×1（変化なし） | 2 | 993 |
| B: docstring を「1つだけ入れる」→「まとめて入れる（例: "LoRA QLoRA"）」 | LoRAとQLoRAの違いは？ | search_glossary×1（`'LoRA QLoRA'`） | 2 | 1209 |
| 元のまま | LoRA、QLoRA、PEFTの関係を教えて | search_glossary×3（1語ずつ） | 2 | 1735 |
| B: まとめて入れる | 同上 | search_glossary×1（`'LoRA QLoRA PEFT'`） | 2 | 1287 |
| C: 「回答の最後に、関係する回を示す」→「回答の最後に、内容の確認問題を1問出す」 | LoRAとQLoRAの違いは？ | search_glossary×2 → make_quiz(2) | 3 | 2641〜2679 |

- 気づき：
  - A は予想と違い、「必ず検索する」の行を消しても、docstring の一文まで消しても、12回すべてで検索してから答えた。ツールが見えていて、質問が講座の用語だと、モデルは自分から検索を選ぶようだった。逆に言うと、この質問では「必ず」の一文が効いているかどうかは確かめられなかった。講座と関係の薄い一般的な質問だと差が出るかもしれない。
  - B は docstring の一語でツールの呼ばれ方がはっきり変わった。検索が1回にまとまり、ツール結果が重複しない分だけ入力トークンが約2〜3割減った（1514→1209、1735→1287）。`requests` は2のままだった。ただ、`'LoRA QLoRA'` で両方が見つかったのは、`search_glossary` が空白を消してから「名前が検索語に含まれるか」も見ているからで、たまたまうまくいった面がある。`MAX_RESULTS=3` なので、4語以上をまとめると結果が切れる。docstring を変えるときは、ツールの実装がその使われ方に耐えられるかも一緒に見る必要がある。
  - C は INSTRUCTIONS の1行で、検索結果を受け取ってから `make_quiz` を呼ぶ流れに変わり、`requests` が2→3、入力トークンが約1.75倍になった。make_quiz の前に検索結果が必要なので、まとめて呼べず順番に呼ぶことになる。指示を1行足すだけで呼び出し回数と費用が増えるのが数字で見えた。3回とも回答は確認問題で終わり、「関係する回：第N回」という締めの行はなくなった。回の番号は本文の中で触れていた。

### 実装：save_learning_log に承認を挟む

- やったこと：
  - `app/tools.py` に `save_learning_log(title, content, lecture=None)` を作った。`learning_log/日時-見出し.md` に、見出し・日時・関係する回・本文を Markdown で書く。見出しの `/` や空白はファイル名に使わないよう `-` に置き換える。docstring には「記録や保存をはっきり頼まれたときだけ使う。頼まれていないのに自分から保存しない」と書いた。
  - `app/agent.py` で `agent.tool_plain(tools.save_learning_log, requires_approval=True)` と登録し、`output_type=[str, DeferredToolRequests]` にした。
  - CLI に `run_with_approval` と `ask_approvals` を足した。承認待ちで止まったら、ツール名と引数（title / content / lecture）を表示して `[y/N]` を聞き、`y` なら `True`、それ以外なら `ToolDenied("利用者が保存を許可しませんでした…")` を `DeferredToolResults` に入れ、`message_history=result.all_messages()` と一緒に `run_sync` をもう一度呼ぶ。
  - テスト `tests/test_learning_log.py` を追加した（API キー不要、TestModel 使用）。承認前はファイルができないこと、y で1件できること、n でできないこと、ファイルの中身を確かめる。`uv run pytest -q` は 15 passed。
- 結果（`lectures/03/results/impl_*.txt`）：
  - 「第3回の学習記録を保存して。内容：…」→ `[承認待ち] save_learning_log` で止まり、引数が表示された。y を入力すると `保存しました: learning_log/20261105-134850-第3回の学習記録.md` が返り、ファイルができた（中身は `impl_saved_example.md`）。
  - 「今日の学びをメモとして保存して」→ n を入力すると、ツールの結果が「利用者が保存を許可しませんでした」になり、モデルは「許可が得られなかったため保存をキャンセルしました」と答えた。ファイルは増えなかった。
  - 「Human-in-the-loopとは？」のように保存を頼まない質問では、`search_glossary` だけが呼ばれ、承認の確認は出なかった。
- 気づき：
  - 承認のあとの再実行では、`result.usage` は再実行の分だけ（`requests=1`）になる。最初の実行の分と足さないと、1つの依頼で何回 LLM を呼んだかは分からない。
  - 1回目の試行では、y で保存されたあとの LLM 呼び出しが混雑エラーで失敗した。自分のスクリプトがそれを見て同じ依頼をやり直したので、同じ内容の記録が2つできた。講義の「冪等性がないツール」の例が、手元で本当に起きた。承認を挟んでも、承認のあとに失敗してやり直すと二重に書き込まれる。保存済みかどうかを先に確かめるか、同じ内容なら上書きするようにしておくと防げそう。

### 発展（a）：ADK で作り直す

- やったこと：
  - `adk_agent/glossary_agent/agent.py` と `__init__.py` を作った。`INSTRUCTIONS` と `search_glossary`・`get_lecture_summary`・`make_quiz` は `app/` のものをそのまま import して使い、`app/` は変えていない。モデルは `app.config.MODEL`（`gemini-3.5-flash-lite`）にそろえた。
  - `PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent` で対話して動くことを確かめた（`results/adv_adk_run_transcript.txt`）。
  - `adk run` ではツール呼び出しが見えないので、`scripts/run_lec03_adk_compare.py` で ADK の `InMemoryRunner` を直接動かし、イベントの `function_call` を取り出して、同じ4つの質問を Pydantic AI 版と並べた（`results/adk_compare.json`）。
- 結果：

| 質問 | ADK のツール呼び出し | Pydantic AI のツール呼び出し | LLM 呼び出し（ADK / PAI） |
|---|---|---|---|
| 宿題の「3レベル」とは？ | search_glossary('3レベル') | 同じ | 2 / 2 |
| LoRAとQLoRAの違いは？ | search_glossary('LoRA')、('QLoRA') | 同じ | 2 / 2 |
| 第2回の確認問題を1問出して | make_quiz(2) | 同じ | 2 / 2 |
| 第3回では何を学ぶ？ | get_lecture_summary(3) | 同じ | 2 / 2 |

  - 4問とも、呼ばれたツールと引数、LLM の呼び出し回数がまったく同じだった。回答の中身もほぼ同じで、どちらも最後に関係する回を付けた。違いは言い回しと長さくらいで、ADK のほうが少し短めだった。確認問題の用語は `random.choice` なので毎回変わる。
- 書き方の違い：
  - Pydantic AI は `Agent(instructions=...)` を作ってから `agent.tool_plain(関数)` で1つずつ登録し、モデルは `run_sync(..., model=...)` で実行時に渡す。そのためテストで TestModel に差し替えられる。結果は `result.all_messages()` と `result.usage` で同期的に読める。
  - ADK は `Agent(model=..., name=..., instruction=..., tools=[関数のリスト])` と、モデルもツールも作るときにまとめて渡す。`name` は必須。実行は Runner とセッションを通し、結果はイベントの非同期ストリームで流れてくる。ツール呼び出しも、その中の `function_call` の部分として読む。
  - ADK は関数を渡すだけで docstring と型ヒントからツールの説明を作るので、ツール側のコードは共通で使えた。指示とツールの説明が同じなら、フレームワークを変えてもモデルのふるまいはほとんど変わらなかった。差が出るのはモデルの外側で、`adk run` ならセッションに履歴が残るので続けて質問できる。`app.agent` は1回ごとに終わる。
- 気づき・疑問：
  - 今回の比較で、エージェントの挙動を決めているのは主に「指示とツールの説明」で、フレームワークは実行の流れや観察のしやすさを決めている、と分かった。`--trace` のような観察の道具は、ADK では自分で書く必要があった。
  - 発展（b）の Pi は、Codespaces の中で試すように指示されているが、今回は Codespaces を使わずに作業したので取り組んでいない。
