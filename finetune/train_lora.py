# finetune/train_lora_l40s.py
"""
일반 LoRA 파인튜닝 (L40S 48GB 최적화)
- FP16/BF16 전체 정밀도 (4bit 양자화 제거)
- 더 큰 배치 사이즈
- 더 빠른 학습 속도
- QLoRA 대비 약간 더 높은 정확도
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 설정 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "/workspace/data/nuclear_qa.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-14b-nuclear-lora")

# LoRA 설정 (L40S 최적화)
LORA_R = int(os.getenv("LORA_R", "32"))  # QLoRA보다 높게 (16→32)
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

# 학습 설정 (L40S 48GB 활용)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))  # QLoRA 2 → LoRA 4
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "4"))  # 8→4
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))  # 2048→4096

logger.info("="*80)
logger.info(" Nuclear Safety Fine-tuning (LoRA - L40S Optimized)")
logger.info("="*80)
logger.info(f" Model: {MODEL_NAME}")
logger.info(f" Dataset: {DATASET_PATH}")
logger.info(f" Output: {OUTPUT_DIR}")
logger.info(f"  LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
logger.info(f" Batch: {BATCH_SIZE}, Grad Accum: {GRADIENT_ACCUMULATION}")
logger.info(f" Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
logger.info(f" Precision: BFloat16 (no quantization)")
logger.info(f" GPU: L40S 48GB")
logger.info("="*80)

# ==================== 모델 & 토크나이저 로드 ====================
logger.info(" Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#  전체 정밀도 로드 (양자화 없음)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # 전체 BF16
    device_map="auto",
    trust_remote_code=True,
    use_flash_attention_2=True,  # Flash Attention 2 활성화
)

# Gradient checkpointing (선택적 - 메모리↓, 속도↓)
# L40S 48GB는 충분하므로 끌 수도 있음
USE_GRAD_CHECKPOINT = os.getenv("USE_GRAD_CHECKPOINT", "0") == "1"
if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
    logger.info(" Gradient checkpointing enabled")
else:
    logger.info(" Gradient checkpointing disabled (faster training)")

logger.info(f" Model loaded in BFloat16")
logger.info(f"   Model size: ~28GB VRAM")

# ==================== LoRA 설정 ====================
lora_config = LoraConfig(
    r=LORA_R,  # 32 (QLoRA의 2배)
    lora_alpha=LORA_ALPHA,  # 64
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

logger.info(" LoRA configuration applied")

# ==================== 데이터셋 로드 ====================
logger.info(f" Loading dataset from {DATASET_PATH}...")

dataset = load_dataset('json', data_files=DATASET_PATH)

# Train/Validation split (90/10)
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

logger.info(f" Dataset loaded:")
logger.info(f"   Train: {len(dataset['train'])} samples")
logger.info(f"   Validation: {len(dataset['test'])} samples")

# ==================== 프롬프트 템플릿 ====================
def format_instruction(example):
    """Qwen2.5 Chat 템플릿"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    system_prompt = """당신은 원자력 안전 전문가입니다. 
KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요.
기술적 정확성을 최우선으로 하며, 안전 관련 사항은 특히 신중하게 설명해야 합니다."""

    if input_text:
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}

추가 정보: {input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    return {"text": prompt}

tokenized_dataset = dataset.map(
    format_instruction,
    remove_columns=dataset['train'].column_names
)

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = tokenized_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

logger.info(f" Tokenization completed")

# ==================== 실시간 진행률 콜백 ====================
class ProgressCallback(TrainerCallback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        import time
        self.epoch_start_time = time.time()
        logger.info(f" Epoch {int(state.epoch) + 1}/{self.total_epochs} 시작")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            total_steps = state.max_steps
            progress = (state.global_step / total_steps) * 100
            
            # 현재 loss 가져오기
            current_loss = state.log_history[-1].get('loss', 0) if state.log_history else 0
            
            logger.info(
                f"Epoch {int(state.epoch) + 1}/{self.total_epochs}, "
                f"Step {state.global_step}/{total_steps} "
                f"({progress:.1f}%), Loss: {current_loss:.4f}"
            )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        import time
        epoch_time = time.time() - self.epoch_start_time
        logger.info(
            f" Epoch {int(state.epoch) + 1}/{self.total_epochs} 완료 "
            f"(소요 시간: {epoch_time/60:.1f}분)"
        )

# ==================== Trainer 설정 (L40S 최적화) ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # 배치 설정 (L40S 활용)
    per_device_train_batch_size=BATCH_SIZE,  # 4 (QLoRA는 2)
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,  # 4 (QLoRA는 8)
    # Effective batch = 4 * 4 = 16 (QLoRA는 2*8=16으로 동일)
    
    # 학습 설정
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    # 최적화 (L40S BF16 최적화)
    fp16=False,
    bf16=True,  # BFloat16 활용
    optim="adamw_torch_fused",  # Fused AdamW (더 빠름)
    
    # 로깅
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="tensorboard",
    logging_first_step=True,
    
    # 저장
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 성능 최적화
    dataloader_num_workers=4,  # 데이터 로딩 병렬화
    dataloader_pin_memory=True,
    group_by_length=True,
    ddp_find_unused_parameters=False,
    
    # L40S 최적화
    tf32=True,  # TF32 활성화 (Ampere/Ada)
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[ProgressCallback(NUM_EPOCHS)]
)

# ==================== 학습 시작 ====================
logger.info("="*80)
logger.info(" Starting training...")
logger.info(f"   Expected time: ~1-2 hours (L40S)")
logger.info(f"   Memory usage: ~35GB / 48GB")
logger.info("="*80)

try:
    trainer.train()
    logger.info("="*80)
    logger.info(" Training completed successfully!")
    logger.info("="*80)
    
    # LoRA 어댑터 저장
    logger.info(f" Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 최종 평가
    logger.info(" Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"   Final eval loss: {eval_results['eval_loss']:.4f}")
    
    logger.info("="*80)
    logger.info(" Fine-tuning completed!")
    logger.info(f" Output saved to: {OUTPUT_DIR}")
    logger.info("="*80)
    
except Exception as e:
    logger.error("="*80)
    logger.error(f" Training failed: {e}")
    logger.error("="*80)
    raise