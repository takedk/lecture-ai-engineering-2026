"""第2回 演習・宿題（基礎・実装・発展）の学習・評価を実行するスクリプト。

ノートブック `notebooks/lora_format_tuning.ipynb` と同じモデル・データ分割・採点関数を用い、
以下の4実験を順に実行して結果を `lectures/02/` 配下に保存する。

1. 演習（標準設定）: 架空の講座 FAQ（5項目）、`num_train_epochs=20`（40ステップ）
2. 宿題・基礎: `CONFIG` の `num_train_epochs` を `5`（10ステップ）に変えて再学習し、標準設定（20エポック）と比較
3. 宿題・実装: `FAQ`・`GROUNDED`・`UNGROUNDED` を独自の題材（社内 AI 開発サンドボックス環境の利用規約 FAQ 5項目）に置き換えて学習前後を比較
4. 宿題・発展: `scripts/make_sft_dataset.py` で生成した用語集 Q&A データ（116件）を `SFTTrainer` で学習させ、
   3種類の評価質問（用語集にある語・言い回しを変えた質問・用語集にない語）で学習前後および `app.tools.search_glossary` と比較
"""

from __future__ import annotations

import copy
import gc
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
import peft
import trl
import trl.trainer.sft_trainer as sft_mod
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from app.tools import search_glossary

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "lectures" / "02"
RES_DIR = OUT_DIR / "results"


class FastChunkedCE(torch.autograd.Function):
    """大語彙（248,320トークン）の lm_head 交差エントロピーを省メモリで計算する関数。"""

    @staticmethod
    def forward(ctx, hidden, w, labels, chunk_size, denom):
        w_dtype = w.to(hidden.dtype)
        grad_h = torch.zeros_like(hidden)
        total_loss = hidden.new_zeros((), dtype=torch.float32)
        total_correct = hidden.new_zeros((), dtype=torch.float32)
        n_valid = int((labels != -100).sum().item())
        scale = 1.0 / float(denom)
        for start in range(0, max(n_valid, 1), chunk_size):
            end = min(start + chunk_size, n_valid)
            if end <= start:
                break
            h_c = hidden[start:end]
            l_c = labels[start:end]
            logits = F.linear(h_c, w_dtype)
            with torch.enable_grad():
                logits_leaf = logits.detach().requires_grad_(True)
                loss_c = F.cross_entropy(logits_leaf.float(), l_c, reduction="sum")
                loss_c.backward()
            total_loss = total_loss + loss_c.detach()
            with torch.no_grad():
                total_correct = total_correct + (logits.argmax(dim=-1) == l_c).sum().float()
                grad_h[start:end] = (logits_leaf.grad * scale).to(hidden.dtype) @ w_dtype
        ctx.save_for_backward(grad_h)
        return total_loss * scale, total_correct

    @staticmethod
    def backward(ctx, grad_loss, grad_correct):
        (grad_h,) = ctx.saved_tensors
        return grad_h * grad_loss.to(grad_h.dtype), None, None, None, None


def fast_chunked_cross_entropy_loss(
    hidden_states,
    lm_head_weight,
    chunk_size,
    labels=None,
    shift_labels=None,
    num_items_in_batch=None,
    logit_scale=1.0,
    final_logit_softcapping=None,
    lm_head_bias=None,
):
    if shift_labels is not None:
        hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        labels = shift_labels.reshape(-1)
    else:
        hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
        labels = labels[..., 1:].reshape(-1)
    valid = labels != -100
    n_valid_tensor = valid.sum()
    order = valid.to(torch.int8).argsort(descending=True, stable=True)
    hidden_packed = hidden[order]
    labels_packed = labels[order]
    denom = num_items_in_batch if num_items_in_batch is not None else n_valid_tensor.clamp(min=1)
    denom_val = denom.item() if isinstance(denom, torch.Tensor) else float(denom)
    loss, correct = FastChunkedCE.apply(hidden_packed, lm_head_weight, labels_packed, 256, max(denom_val, 1.0))
    entropy_sum = hidden.new_zeros((), dtype=torch.float32)
    return loss, correct, entropy_sum, n_valid_tensor


BASE_CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 20,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 96,
    "gen": {"do_sample": False},
}

SYSTEM = (
    "あなたは講座 FAQ の回答係です。与えられた FAQ だけを根拠に、必ず JSON 1 行で答えてください。"
    '形式: {"answer": 文字列, "needs_review": 真偽値}。'
    "FAQ に記載が無いことは answer に『確認が必要』と書き、needs_review を true にします。"
)
UNGROUNDED_ANSWER = "この FAQ には記載がありません。運営に確認が必要です。"


def check_format(text: str):
    """JSON として読め、必須キーと型が合うか。失敗理由も返す。"""
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_error: {type(e).__name__}", None
    if not isinstance(obj, dict):
        return False, "not_object", None
    if set(obj) - {"answer", "needs_review"}:
        return False, "extra_keys", obj
    if "answer" not in obj or "needs_review" not in obj:
        return False, "missing_key", obj
    if not isinstance(obj["answer"], str) or not isinstance(obj["needs_review"], bool):
        return False, "wrong_type", obj
    return True, "", obj


def check_content(obj, gold_entry):
    """内容の合否: needs_review が正解と一致し、必須語句を answer に含む。"""
    if obj is None:
        return False
    if obj.get("needs_review") != gold_entry["needs_review"]:
        return False
    return all(w in obj.get("answer", "") for w in gold_entry["must_include"])


def classify(raw: str) -> str:
    ok, _, obj = check_format(raw)
    if not ok or obj is None:
        return "format_error"
    return "needs_review" if obj["needs_review"] else "asserted"


# 宿題（実装）で用意した独自 FAQ（社内 AI 開発サンドボックス環境の利用規約 5項目）
IMPL_FAQ = {
    "gpu_quota": "GPU インスタンスの連続利用上限は 1 回あたり 4 時間、月間合計 40 時間までです。",
    "storage": "作業用データの保存先は /work/project ディレクトリで、容量上限は 1 人 50GB です。",
    "data_policy": "社外秘データや顧客の個人情報は、サンドボックス環境へアップロードしてはいけません。",
    "model_weights": "商用利用が許可されたオープンウェイトモデル（Apache-2.0 または MIT ライセンス）のみ利用可能です。",
    "support": "環境トラブルの問い合わせ先は Slack の #ai-sandbox-help チャンネル（平日 10:00〜18:00 対応）です。",
}

IMPL_GROUNDED = {
    "gpu_quota": (
        [
            "GPU インスタンスは何時間まで連続で使えますか",
            "GPU の利用時間の上限を教えてください",
            "月間の GPU 利用枠は何時間ですか",
            "1 回の GPU 利用時間の制限は？",
        ],
        ["4 時間", "40 時間"],
    ),
    "storage": (
        [
            "作業用データはどこに保存すればよいですか",
            "ストレージの容量上限は何 GB ですか",
            "データの保存先ディレクトリを教えてください",
            "保存できるデータ量の上限は？",
        ],
        ["/work/project", "50GB"],
    ),
    "data_policy": (
        [
            "顧客の個人情報をアップロードして分析してもよいですか",
            "社外秘データの持ち込みルールを教えてください",
            "機密データを環境に入れても大丈夫ですか",
            "個人情報のアップロードは可能ですか",
        ],
        ["社外秘", "個人情報"],
    ),
    "model_weights": (
        [
            "どのライセンスのオープンウェイトモデルが使えますか",
            "利用できるモデルのライセンス条件を教えてください",
            "Apache-2.0 のモデルは利用できますか",
            "使えるモデルの条件は？",
        ],
        ["Apache-2.0", "MIT"],
    ),
    "support": (
        [
            "環境トラブルが起きたらどこに連絡すればよいですか",
            "サポート窓口の Slack チャンネルと対応時間を教えてください",
            "不具合の問い合わせ先はどこですか",
            "トラブル時の相談先は？",
        ],
        ["#ai-sandbox-help", "10:00〜18:00"],
    ),
}

IMPL_UNGROUNDED = [
    "GPU インスタンスを 4 時間以上続けて使いたい場合は延長申請できますか",
    "複数人で同じプロジェクトディレクトリを共有することはできますか",
    "保存先の 50GB を超えた場合に容量を追加申請することはできますか",
    "学習させた LoRA アダプタを外部の Hugging Face Hub に公開してもよいですか",
    "Llama ライセンスや Gemma ライセンスのモデルは特例で使えますか",
    "Docker イメージを自分でビルドして持ち込むことはできますか",
    "休日や夜間に環境が停止した場合の緊急連絡先はありますか",
    "外部の有料 LLM API のキーを環境変数に設定して呼び出せますか",
]

IMPL_TEST_EXTRA_GROUNDED = {
    "gpu_quota": ["GPU の連続利用時間と月間上限を教えてください", "1 回あたりと月間の GPU 利用上限はそれぞれ何時間ですか"],
    "storage": ["データの保存場所と 1 人あたりの容量上限を教えてください", "作業ファイルの保存先パスと上限サイズは何ですか"],
    "data_policy": ["社内機密や顧客データをサンドボックスに置いてもよいですか", "アップロードが禁止されているデータの種類を教えてください"],
    "model_weights": ["サンドボックス環境で許可されているモデルのライセンスは何ですか", "MIT ライセンスや Apache-2.0 のモデルは使えますか"],
    "support": ["Slack のサポートチャンネル名と受付時間を教えてください", "環境トラブルの質問は何時から何時までどこで受け付けていますか"],
}

IMPL_TEST_EXTRA_UNGROUNDED = [
    "退職や異動の際に /work/project のデータはいつ削除されますか",
    "GPU の種類（A100 や L4 など）を自分で選ぶことはできますか",
    "月間 40 時間の利用枠を翌月に繰り越すことはできますか",
    "OS パッケージの追加インストールに sudo 権限は使えますか",
]

# 宿題（発展）の評価セット（15問：A 用語集にある語 5問 / B 言い回し・別名を変えた質問 5問 / C 用語集にない語 5問）
ADV_EVAL_SET = [
    {"id": "A1", "category": "A_in_glossary", "question": "「AIエンジニアリング」とは何ですか？", "in_glossary": True, "must_include": ["基盤モデル", "評価"], "any_include": ["アプリケーション", "システム", "サイクル"]},
    {"id": "A2", "category": "A_in_glossary", "question": "「LoRA」とは何ですか？", "in_glossary": True, "must_include": ["固定"], "any_include": ["低ランク", "行列", "アダプタ", "PEFT"]},
    {"id": "A3", "category": "A_in_glossary", "question": "「ハルシネーション」とは何ですか？", "in_glossary": True, "must_include": ["事実と異なる"], "any_include": ["もっともらし", "生成", "幻覚", "存在しない"]},
    {"id": "A4", "category": "A_in_glossary", "question": "「Pydantic AI」とは何ですか？", "in_glossary": True, "must_include": ["エージェント"], "any_include": ["フレームワーク", "Pydantic", "型", "TestModel"]},
    {"id": "A5", "category": "A_in_glossary", "question": "「宿題の3レベル」とは何ですか？", "in_glossary": True, "must_include": ["基礎", "実装", "発展"], "any_include": ["どれか1つ", "完了", "採点"]},
    {"id": "B1", "category": "B_paraphrased", "question": "Low-Rank Adaptation とはどのような手法ですか？", "in_glossary": True, "must_include": ["行列"], "any_include": ["固定", "低ランク", "差分", "アダプタ"]},
    {"id": "B2", "category": "B_paraphrased", "question": "HITL（人間による承認）はエージェントのどんな操作の前に使いますか？", "in_glossary": True, "must_include": ["確認", "承認"], "any_include": ["取り消しにくい", "書き込み", "送信", "requires_approval"]},
    {"id": "B3", "category": "B_paraphrased", "question": "この講座のデフォルトモデルには何が選ばれていますか？", "in_glossary": True, "must_include": ["gemini-3.5-flash-lite"], "any_include": ["無料枠", "レート制限", "AIE_MODEL"]},
    {"id": "B4", "category": "B_paraphrased", "question": "コーディングエージェントの「Pi」にはどんな特徴がありますか？", "in_glossary": True, "must_include": ["ターミナル"], "any_include": ["オープンソース", "MIT", "最小限", "read", "bash"]},
    {"id": "B5", "category": "B_paraphrased", "question": "MCPのトランスポートである stdio と Streamable HTTP はどう違いますか？", "in_glossary": True, "must_include": ["サブプロセス", "URL"], "any_include": ["標準入出力", "HTTP", "認証"]},
    {"id": "C1", "category": "C_out_of_glossary", "question": "「DPO（Direct Preference Optimization）」とは何ですか？", "in_glossary": False, "is_fictional_course_term": False, "must_include": [], "any_include": ["選好", "強化学習", "Preference", "報酬"]},
    {"id": "C2", "category": "C_out_of_glossary", "question": "「vLLM」とは何ですか？", "in_glossary": False, "is_fictional_course_term": False, "must_include": [], "any_include": ["推論", "PagedAttention", "高速", "サービング"]},
    {"id": "C3", "category": "C_out_of_glossary", "question": "「FlashAttention」とは何ですか？", "in_glossary": False, "is_fictional_course_term": False, "must_include": [], "any_include": ["注意", "Attention", "メモリ", "高速", "GPU"]},
    {"id": "C4", "category": "C_out_of_glossary", "question": "「lec99-final タグ」とは何ですか？", "in_glossary": False, "is_fictional_course_term": True, "must_include": [], "any_include": []},
    {"id": "C5", "category": "C_out_of_glossary", "question": "「AIE2026の修了試験の合格点」とは何ですか？", "in_glossary": False, "is_fictional_course_term": True, "must_include": [], "any_include": []},
]
