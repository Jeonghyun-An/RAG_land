# finetune/train_lora_l40s.py 를 여기다 옮긴거임
"""
일반 LoRA 파인튜닝 (L40S 48GB 최적화) - STABLE FIX
- FlashAttention2 강제 비활성화 (flash_attn 미설치여도 OK)
- assistant-only labels 마스킹을 "토큰 ID 시퀀스" 검색으로 안정화 (decode/re-tokenize 제거)
- 커스텀 collator로 labels를 -100 패딩 (배치 텐서화 안정)
"""

import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ENV ====================
MODEL_NAME = os.getenv("FT_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "/workspace/data/nuclear_qa.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-14b-nuclear-lora")

# LoRA
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

# Train
BATCH_SIZE = int(os.getenv("FT_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
FT_MAX_SEQ_LENGTH = int(os.getenv("FT_MAX_SEQ_LENGTH", "4096"))

# Options
USE_GRAD_CHECKPOINT = os.getenv("USE_GRAD_CHECKPOINT", "0") == "1"
PIN_GPU0 = os.getenv("PIN_GPU0", "1") == "1"
TRAIN_ASSISTANT_ONLY = os.getenv("TRAIN_ASSISTANT_ONLY", "1") == "1"

# 안정성 위해 기본 0 권장 (필요하면 2~4로 올리기)
DATALOADER_NUM_WORKERS = int(os.getenv("DATALOADER_NUM_WORKERS", "0"))
OPTIM = os.getenv("OPTIM", "adamw_torch_fused")

# ==================== LOG ====================
logger.info("=" * 80)
logger.info(" Nuclear Safety Fine-tuning (LoRA - L40S Optimized) [STABLE FIX]")
logger.info("=" * 80)
logger.info(f" Model: {MODEL_NAME}")
logger.info(f" Dataset: {DATASET_PATH}")
logger.info(f" Output: {OUTPUT_DIR}")
logger.info(f"  LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
logger.info(f" Batch: {BATCH_SIZE}, Grad Accum: {GRADIENT_ACCUMULATION}")
logger.info(f" Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
logger.info(f" Precision: BF16 (no quantization)")
logger.info(f" Max seq length: {FT_MAX_SEQ_LENGTH}")
logger.info(f" Train assistant only: {TRAIN_ASSISTANT_ONLY}")
logger.info(f" Dataloader workers: {DATALOADER_NUM_WORKERS}")
logger.info("=" * 80)

# ==================== TOKENIZER ====================
logger.info(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# assistant marker token ids (토큰 ID 시퀀스로 직접 찾을 것)
# Qwen 계열은 보통 "<|im_start|>assistant\n" 형태가 들어감
ASSISTANT_MARKER_TEXT = "<|im_start|>assistant"
assistant_marker_ids = tokenizer.encode(ASSISTANT_MARKER_TEXT, add_special_tokens=False)
if len(assistant_marker_ids) == 0:
    logger.warning("assistant marker ids empty - assistant-only masking may fallback to full labels.")

# ==================== CONFIG (disable FlashAttention2) ====================
logger.info(" Loading config (disable FlashAttention2)...")
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

for attr in ["use_flash_attention_2", "flash_attn_2_enabled", "_flash_attn_2_enabled"]:
    if hasattr(config, attr):
        try:
            setattr(config, attr, False)
        except Exception:
            pass

try:
    config.attn_implementation = "sdpa"
except Exception:
    pass
try:
    config._attn_implementation = "sdpa"
except Exception:
    pass

# ==================== MODEL ====================
logger.info(" Loading model (BF16, SDPA)...")
device_map = {"": 0} if PIN_GPU0 else "auto"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model.config.pad_token_id = tokenizer.pad_token_id

if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
    logger.info(" Gradient checkpointing enabled")
else:
    logger.info(" Gradient checkpointing disabled")

# ==================== LoRA ====================
logger.info(" Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== DATASET ====================
logger.info(f" Loading dataset from {DATASET_PATH}...")
ds = load_dataset("json", data_files=DATASET_PATH)
raw = ds["train"]

for k in ["instruction", "output"]:
    if k not in raw.column_names:
        raise ValueError(f"Dataset missing required field '{k}'. Found columns: {raw.column_names}")

split = raw.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

logger.info(f" Train: {len(train_ds)} / Eval: {len(eval_ds)}")

# ==================== PROMPT ====================
SYSTEM_PROMPT = (
    "당신은 원자력 안전 전문가입니다.\n"
    "KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요.\n"
    "기술적 정확성을 최우선으로 하며, 안전 관련 사항은 특히 신중하게 설명해야 합니다."
)

def _apply_chat(messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # fallback
    def wrap(role: str, content: str) -> str:
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"
    out = ""
    for m in messages:
        out += wrap(m["role"], m["content"])
    return out

def build_user(ex: Dict[str, Any]) -> str:
    instruction = (ex.get("instruction") or "").strip()
    input_text = (ex.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\n추가 정보: {input_text}".strip()
    return instruction

def to_text(ex: Dict[str, Any]) -> Dict[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(ex)},
        {"role": "assistant", "content": (ex.get("output") or "").strip()},
    ]
    return {"text": _apply_chat(messages)}

logger.info(" Formatting dataset...")
train_text = train_ds.map(to_text, remove_columns=train_ds.column_names)
eval_text = eval_ds.map(to_text, remove_columns=eval_ds.column_names)

# ==================== ASSISTANT START FINDER (TOKEN ID SEARCH) ====================
def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return start index of needle in haystack, or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    max_i = len(haystack) - len(needle)
    for i in range(max_i + 1):
        if haystack[i] != first:
            continue
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1

def mask_labels_assistant_only(input_ids: List[int]) -> List[int]:
    """
    labels length MUST equal input_ids length.
    assistant marker 위치를 토큰 id로 찾아, 그 전은 -100.
    """
    if not TRAIN_ASSISTANT_ONLY or not assistant_marker_ids:
        return input_ids.copy()

    pos = find_subsequence(input_ids, assistant_marker_ids)
    if pos < 0:
        # marker 못 찾으면 전체 학습으로 fallback
        return input_ids.copy()

    # marker 자체부터 학습시켜도 되지만 보통 marker 이후만 학습시키는 편이 좋음
    start = pos + len(assistant_marker_ids)
    return ([-100] * start) + input_ids[start:].copy()

# ==================== TOKENIZE ====================
def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=FT_MAX_SEQ_LENGTH,
        padding=False,
    )

    labels = []
    for ids in enc["input_ids"]:
        lab = mask_labels_assistant_only(ids)

        # 100% 안전장치: 혹시라도 길이 안 맞으면 전체학습으로 덮어씀
        if len(lab) != len(ids):
            lab = ids.copy()
        labels.append(lab)

    enc["labels"] = labels
    return enc

logger.info(" Tokenizing...")
train_tok = train_text.map(tokenize_batch, batched=True, remove_columns=["text"])
eval_tok = eval_text.map(tokenize_batch, batched=True, remove_columns=["text"])
logger.info(" Tokenization completed")

# ==================== CUSTOM COLLATOR (labels safe pad) ====================
class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) tokenizer.pad가 labels를 텐서화하다가 터지므로 labels는 먼저 분리
        labels = [f.pop("labels") for f in features]  # <-- 핵심 (pop)

        # 2) input_ids/attention_mask 등만 패딩해서 텐서화
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # 3) labels는 우리가 -100으로 직접 패딩해서 텐서화
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + ([-100] * (max_len - len(lab)))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ==================== PROGRESS ====================
class ProgressCallback(TrainerCallback):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        logger.info(f" Epoch {current_epoch}/{self.total_epochs} 시작")

# ==================== TRAIN ARGS ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=False,
    bf16=True,
    optim=OPTIM,
    logging_steps=10,
    report_to="tensorboard",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=False,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,
    dataloader_pin_memory=True,
    group_by_length=True,
    tf32=True,
    remove_unused_columns=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=CausalLMCollator(tokenizer, pad_to_multiple_of=8),
    callbacks=[ProgressCallback(NUM_EPOCHS)],
)

# ==================== RUN ====================
logger.info("=" * 80)
logger.info(" Starting training...")
logger.info("=" * 80)

trainer.train()

logger.info(" Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info(" Final evaluation...")
res = trainer.evaluate()
logger.info(f" eval_loss: {res.get('eval_loss')}")

logger.info(" Done.")
