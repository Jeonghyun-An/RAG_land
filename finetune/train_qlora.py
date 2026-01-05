# finetune/train_qlora.py
"""
QLoRA 파인튜닝 스크립트
- 4bit 양자화로 메모리 효율 최적화
- 실시간 진행률 로깅
- 원자력 도메인 특화 프롬프트 템플릿
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 설정 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "/workspace/data/nuclear_qa.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/output/qwen2.5-14b-nuclear-lora")

# QLoRA 설정
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

# 학습 설정
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "8"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

logger.info("="*80)
logger.info(" Nuclear Safety Fine-tuning (QLoRA)")
logger.info("="*80)
logger.info(f" Model: {MODEL_NAME}")
logger.info(f" Dataset: {DATASET_PATH}")
logger.info(f" Output: {OUTPUT_DIR}")
logger.info(f"  LoRA Config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
logger.info(f" Batch: {BATCH_SIZE}, Grad Accum: {GRADIENT_ACCUMULATION}")
logger.info(f" Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
logger.info("="*80)

# ==================== 4bit 양자화 설정 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ==================== 모델 & 토크나이저 로드 ====================
logger.info(" Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Gradient checkpointing (메모리 절약)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

logger.info(" Model and tokenizer loaded")

# ==================== LoRA 설정 ====================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],  # Qwen2.5 attention + MLP 모듈
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
    """
    Qwen2.5 Chat 템플릿 적용
    원자력 안전 전문가 페르소나 설정
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # 시스템 프롬프트
    system_prompt = """당신은 원자력 안전 전문가입니다. 
KINAC 규정과 IAEA 가이드라인에 기반하여 정확하고 상세한 답변을 제공하세요.
기술적 정확성을 최우선으로 하며, 안전 관련 사항은 특히 신중하게 설명해야 합니다."""

    # 프롬프트 구성
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

# 데이터셋 포매팅
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
    """실시간 진행률 로깅 콜백"""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f" Epoch {int(state.epoch) + 1}/{self.total_epochs} 시작")
        
    def on_step_end(self, args, state, control, **kwargs):
        # 10 스텝마다 진행률 출력
        if state.global_step % 10 == 0:
            total_steps = state.max_steps
            progress = (state.global_step / total_steps) * 100
            logger.info(
                f"Epoch {int(state.epoch) + 1}/{self.total_epochs}, "
                f"Step {state.global_step}/{total_steps} "
                f"({progress:.1f}%)"
            )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f" Epoch {int(state.epoch) + 1}/{self.total_epochs} 완료")

# ==================== Trainer 설정 ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # 배치 설정
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    # 학습 설정
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    # 최적화
    fp16=False,
    bf16=True,  # bfloat16 권장
    optim="paged_adamw_8bit",  # 메모리 효율적 옵티마이저
    
    # 로깅 (진행률 추적용)
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="tensorboard",
    logging_first_step=True,  # 첫 스텝부터 로깅
    
    # 저장
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 기타
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    group_by_length=True,  # 길이별 그룹화로 효율 향상
    ddp_find_unused_parameters=False,
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
    
    logger.info("="*80)
    logger.info(" Fine-tuning completed!")
    logger.info(f" Output saved to: {OUTPUT_DIR}")
    logger.info("="*80)
    
except Exception as e:
    logger.error("="*80)
    logger.error(f" Training failed: {e}")
    logger.error("="*80)
    raise