# /workspace/finetune/train_lora_l40s.py
"""
QLoRA Fine-tuning (L40S 48GB friendly) for Qwen/Qwen2.5-14B-Instruct
- Fixes label length mismatch by custom collator (pads labels to -100)
- Uses BitsAndBytesConfig ONLY (no load_in_4bit kwarg conflict)
- Memory-safe defaults: seq 2048, batch 1, grad checkpoint ON
"""

import os
import logging
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_qlora")

# -------------------- Config (Env) --------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "/workspace/data/nuclear_qa.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-7b-qlora")

# IMPORTANT: safe defaults (pipeline가 4096/2/0으로 던져서 터짐)
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "16"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))

LORA_R = int(os.getenv("LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

USE_GRAD_CHECKPOINT = os.getenv("USE_GRAD_CHECKPOINT", "1") == "1"
OPTIM = os.getenv("OPTIM", "paged_adamw_8bit")

LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "10"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "200"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "200"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a nuclear safety expert. Answer accurately based on KINAC regulations and IAEA guidelines.",
)

# fragmentation 완화 (OOM “한 끗” 방지)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

logger.info("=" * 88)
logger.info("QLoRA Fine-tuning (L40S friendly)")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Dataset: {DATASET_PATH}")
logger.info(f"Output: {OUTPUT_DIR}")
logger.info(f"Seq: {MAX_SEQ_LENGTH} | Batch: {BATCH_SIZE} | GradAcc: {GRADIENT_ACCUMULATION}")
logger.info(f"Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE}")
logger.info(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
logger.info(f"Grad checkpoint: {USE_GRAD_CHECKPOINT} | Optim: {OPTIM}")
logger.info("=" * 88)

# -------------------- Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------- Model (4-bit QLoRA) --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,   # <-- 이것만 사용 (load_in_4bit kwarg 금지)
    attn_implementation="sdpa",       # flash_attn 없어도 OK
)

model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=USE_GRAD_CHECKPOINT,
)

if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

# -------------------- LoRA --------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)

try:
    model.print_trainable_parameters()
except Exception:
    pass

# -------------------- Dataset --------------------
ds = load_dataset("json", data_files=DATASET_PATH)
raw = ds["train"]

for k in ["instruction", "output"]:
    if k not in raw.column_names:
        raise ValueError(f"Dataset missing required field '{k}'. Found columns: {raw.column_names}")

split = raw.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds = split["test"]
logger.info(f"Loaded dataset | train={len(train_ds)} eval={len(eval_ds)}")

# -------------------- Prompt formatting --------------------
def _user_content(ex: Dict[str, Any]) -> str:
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    if inp:
        return f"{inst}\n\nAdditional context:\n{inp}".strip()
    return inst

def format_chat(ex: Dict[str, Any]) -> Dict[str, str]:
    user = _user_content(ex)
    assistant = (ex.get("output") or "").strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_text = train_ds.map(format_chat, remove_columns=train_ds.column_names)
eval_text = eval_ds.map(format_chat, remove_columns=eval_ds.column_names)

# -------------------- Tokenize --------------------
def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    # labels = input_ids (causal LM)
    enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
    return enc

train_tok = train_text.map(tokenize_batch, batched=True, remove_columns=["text"])
eval_tok = eval_text.map(tokenize_batch, batched=True, remove_columns=["text"])
logger.info("Tokenization completed")

# -------------------- ✅ Custom collator (fix label mismatch 100%) --------------------
class CausalLMCollator:
    """
    - Pads input_ids/attention_mask using tokenizer.pad
    - Pads labels to same length with -100
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels first (tokenizer.pad doesn't reliably pad arbitrary fields)
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels", None)

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) > max_len:
                lab = lab[:max_len]
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

data_collator = CausalLMCollator(tokenizer)

# -------------------- Progress callback --------------------
class ProgressCallback(TrainerCallback):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) + 1 if state.epoch is not None else 1
        logger.info(f"[Epoch {ep}/{self.total_epochs}] start")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            logger.info(f"step={state.global_step} loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            logger.info(f"step={state.global_step} eval_loss={logs['eval_loss']:.4f}")

# -------------------- TrainingArguments --------------------
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

    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    report_to="tensorboard",

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=False,

    # ✅ worker에서 pad/labels 변환 에러 줄이려면 0이 제일 안정적
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    group_by_length=True,

    tf32=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
    callbacks=[ProgressCallback(NUM_EPOCHS)],
)

logger.info("=" * 88)
logger.info("Starting training...")
logger.info("=" * 88)

trainer.train()

logger.info("=" * 88)
logger.info("Training completed. Saving adapter + tokenizer...")
logger.info("=" * 88)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Running final evaluation...")
metrics = trainer.evaluate()
logger.info(f"Final eval_loss={metrics.get('eval_loss')}")

logger.info(f"Done. Output saved to: {OUTPUT_DIR}")
