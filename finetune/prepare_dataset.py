# finetune/prepare_dataset.py
import json
import random
from pathlib import Path
from typing import List, Dict

def create_sample_dataset():
    """
    샘플 원자력 안전 QA 데이터셋 생성
    실제로는 Milvus나 CUBRID에서 추출
    """
    
    # 샘플 데이터 (실제로는 수천 개 필요)
    qa_pairs = [
        {
            "instruction": "방사선작업종사자의 연간 유효선량 한도는?",
            "input": "",
            "output": "방사선작업종사자의 유효선량 한도는 연간 50mSv이며, 5년간 누적선량이 100mSv를 초과하지 않아야 합니다. (원자력안전법 시행령 제57조)"
        },
        {
            "instruction": "원자로 냉각재 계통의 주요 구성요소는?",
            "input": "가압경수로(PWR) 기준",
            "output": "원자로 냉각재 계통은 원자로 압력용기, 증기발생기(2~4개), 원자로냉각재펌프(4개), 가압기로 구성됩니다. 1차 계통은 약 15.5 MPa의 고압으로 운전됩니다."
        },
        {
            "instruction": "IAEA Safety Standards 중 SF-1의 목적은?",
            "input": "",
            "output": "SF-1(Fundamental Safety Principles)은 원자력 안전의 기본 원칙을 제시하는 최상위 문서로, 방사선 위험으로부터 사람과 환경을 보호하기 위한 10가지 안전 원칙을 정의합니다."
        },
        {
            "instruction": "원자력발전소의 심층방어 개념은?",
            "input": "",
            "output": "심층방어는 5개 층으로 구성됩니다: 1층-설계 품질 보증, 2층-운전 안전, 3층-설계기준사고 대응, 4층-중대사고 관리, 5층-방사능 누출 완화입니다."
        },
        {
            "instruction": "제어봉의 주요 기능은?",
            "input": "",
            "output": "제어봉은 중성자를 흡수하여 핵분열 반응을 제어합니다. 주요 기능은 반응도 제어, 출력 조절, 긴급 정지(SCRAM) 수행입니다."
        },
        {
            "instruction": "격납건물의 설계 압력은?",
            "input": "APR1400 기준",
            "output": "APR1400의 격납건물 설계압력은 약 392 kPa(absolute)이며, LOCA 시 압력 상승을 견딜 수 있도록 설계됩니다."
        },
        {
            "instruction": "안전주입계통의 목적은?",
            "input": "",
            "output": "안전주입계통은 냉각재 상실사고(LOCA) 시 원자로에 붕산수를 주입하여 노심을 냉각하고 핵분열 반응을 정지시킵니다."
        },
        {
            "instruction": "방사성폐기물의 분류는?",
            "input": "",
            "output": "방사성폐기물은 중·저준위와 고준위로 분류됩니다. 중·저준위는 운전폐기물, 고준위는 사용후핵연료가 대표적입니다."
        },
    ]
    
    # 데이터 증강: 변형 생성
    augmented_data = []
    for qa in qa_pairs:
        # 원본
        augmented_data.append(qa)
        
        # 변형 1: 다른 표현
        if "한도" in qa['instruction']:
            augmented_data.append({
                "instruction": qa['instruction'].replace("한도", "제한값"),
                "input": qa['input'],
                "output": qa['output']
            })
        
        # 변형 2: 축약형
        if len(qa['output']) > 100:
            augmented_data.append({
                "instruction": f"{qa['instruction']} (간단히)",
                "input": qa['input'],
                "output": qa['output'].split('.')[0] + "."
            })
        
        # 변형 3: 상세 질문
        if "주요" in qa['instruction']:
            augmented_data.append({
                "instruction": qa['instruction'].replace("주요", "모든"),
                "input": qa['input'],
                "output": qa['output']
            })
    
    return augmented_data

def save_dataset(data: List[Dict], output_path: str):
    """JSONL 형식으로 저장"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(data)} examples to {output_path}")

def main():
    # Docker 컨테이너 내부에서는 /workspace/data로 마운트됨
    base_path = "/workspace/data"
    
    # 데이터 생성
    dataset = create_sample_dataset()
    
    # Train/Test 분할 (90/10)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # 저장 (Docker 내부 경로)
    save_dataset(train_data, f"{base_path}/nuclear_qa.jsonl")
    save_dataset(test_data, f"{base_path}/test_qa.jsonl")
    
    print(f"📊 Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"💾 Files saved to: {base_path}")

if __name__ == "__main__":
    main()