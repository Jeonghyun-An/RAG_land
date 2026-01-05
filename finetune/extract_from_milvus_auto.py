#!/usr/bin/env python3
# finetune/extract_from_milvus_auto.py
"""
Automatic language detection and QA generation for both Korean and English documents

Usage:
    python finetune/extract_from_milvus_auto.py --doc-ids DOC1 DOC2 --output data/training_data_mixed.jsonl

Features:
    - Automatic Korean/English detection
    - Language-specific QA patterns
    - Mixed-language dataset support
    - Separate output files per language (optional)
"""
import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from pymilvus import connections, Collection

# Milvus settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

# Korean patterns (from original)
QA_PATTERNS_KO = {
    "law": {
        "keywords": ["제", "조", "항", "호", "법", "규정", "기준"],
        "templates": [
            "{topic}의 내용은 무엇인가요?",
            "{topic}에 대해 설명해주세요.",
            "{topic}에 관한 법적 규정은?",
        ]
    },
    "manual": {
        "keywords": ["절차", "방법", "단계", "수행", "작업"],
        "templates": [
            "{topic}의 절차를 설명해주세요.",
            "{topic}은 어떻게 수행하나요?",
            "{topic} 시 주의사항은?",
        ]
    },
    "technical": {
        "keywords": ["설계", "계통", "구조", "원리", "기능"],
        "templates": [
            "{topic}의 원리를 설명해주세요.",
            "{topic}의 주요 기능은?",
            "{topic}은 어떻게 작동하나요?",
        ]
    },
    "safety": {
        "keywords": ["안전", "사고", "대응", "비상", "방호"],
        "templates": [
            "{topic} 발생 시 대응 절차는?",
            "{topic}을 위한 안전 조치는?",
            "{topic}의 안전 기준은?",
        ]
    }
}

# English patterns
QA_PATTERNS_EN = {
    "iaea_safety": {
        "keywords": ["iaea", "safety standard", "requirement", "shall"],
        "templates": [
            "What are the IAEA requirements for {topic}?",
            "Explain the safety standards for {topic}.",
            "What does IAEA say about {topic}?",
        ]
    },
    "technical": {
        "keywords": ["system", "component", "design", "function"],
        "templates": [
            "Describe the {topic}.",
            "What is the function of {topic}?",
            "How does {topic} work?",
        ]
    },
    "procedure": {
        "keywords": ["procedure", "method", "step", "process"],
        "templates": [
            "What is the procedure for {topic}?",
            "How to perform {topic}?",
            "Describe the steps for {topic}.",
        ]
    },
    "safety_principle": {
        "keywords": ["defence in depth", "redundancy", "diversity"],
        "templates": [
            "Explain the principle of {topic}.",
            "How is {topic} applied in nuclear safety?",
        ]
    }
}


def detect_language(text: str) -> str:
    """
    Detect language of text (Korean or English)
    
    Returns: 'ko', 'en', or 'mixed'
    """
    # Count Korean characters (Hangul)
    korean_chars = len(re.findall(r'[가-힣]', text))
    # Count English characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = korean_chars + english_chars
    if total == 0:
        return 'unknown'
    
    korean_ratio = korean_chars / total
    
    if korean_ratio > 0.7:
        return 'ko'
    elif korean_ratio < 0.3:
        return 'en'
    else:
        return 'mixed'


def connect_milvus():
    """Connect to Milvus"""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f" Connected to Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f" Milvus connection failed: {e}")
        raise


def extract_chunks_by_doc(doc_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Extract all chunks for a specific document"""
    try:
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        results = collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )

        print(f"   {doc_id}: {len(results)} chunks extracted")
        return results

    except Exception as e:
        print(f"   Failed to extract {doc_id}: {e}")
        return []


def classify_chunk_type(chunk: Dict[str, Any], language: str) -> str:
    """Classify chunk type based on language"""
    text = chunk.get('chunk', '').lower()
    
    if language == 'ko':
        patterns = QA_PATTERNS_KO
    else:  # en or mixed
        patterns = QA_PATTERNS_EN
    
    for chunk_type, config in patterns.items():
        if any(kw in text for kw in config["keywords"]):
            return chunk_type
    
    return "general"


def extract_topic(chunk: Dict[str, Any], language: str) -> str:
    """Extract topic based on language"""
    text = chunk.get('chunk', '')
    section = chunk.get('section', '')
    
    # Use section if available
    if section and section not in ["Unknown", "META"]:
        return section.strip()
    
    if language == 'ko':
        # Korean: Extract from first sentence
        first_sentence = text.split('.')[0].split('\n')[0]
        if len(first_sentence) > 10 and len(first_sentence) < 100:
            return first_sentence.strip()
        return "관련 내용"
    
    else:  # English
        # Extract from first sentence or heading
        match = re.search(r'^([A-Z][A-Za-z\s]{5,50})', text)
        if match:
            return match.group(1).strip()
        return "this topic"


def generate_qa_samples(
    chunks: List[Dict[str, Any]],
    doc_id: str,
    augment: bool = True,
    separate_by_lang: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate QA samples with language detection
    
    Returns:
        {
            'ko': [samples],
            'en': [samples],
            'mixed': [samples]
        }
    """
    samples_by_lang = {'ko': [], 'en': [], 'mixed': []}

    for chunk in chunks:
        text = chunk.get('chunk', '').strip()
        if not text or len(text) < 50:
            continue

        # Detect language
        language = detect_language(text)
        
        if language == 'unknown':
            continue

        # Remove meta lines
        text = re.sub(r'^META:.*?\n', '', text)

        chunk_type = classify_chunk_type(chunk, language)
        topic = extract_topic(chunk, language)
        section = chunk.get('section', '')
        page = chunk.get('page', 0)

        # Select templates based on language
        if language == 'ko':
            patterns = QA_PATTERNS_KO
        else:
            patterns = QA_PATTERNS_EN

        if chunk_type in patterns:
            templates = patterns[chunk_type]["templates"]
        else:
            if language == 'ko':
                templates = ["{topic}에 대해 설명해주세요.", "{topic}은 무엇인가요?"]
            else:
                templates = ["Explain {topic}.", "What is {topic}?"]

        # Generate questions
        for template in templates[:2]:
            instruction = template.format(topic=topic)

            # Build input context
            input_parts = []
            if doc_id:
                input_parts.append(f"Document: {doc_id}" if language == 'en' else f"문서: {doc_id}")
            if section and section not in ["Unknown", "META"]:
                input_parts.append(f"Section: {section}" if language == 'en' else f"섹션: {section}")
            if page:
                input_parts.append(f"Page: {page}" if language == 'en' else f"페이지: {page}")

            input_text = ", ".join(input_parts)

            sample = {
                "instruction": instruction,
                "input": input_text,
                "output": text,
                "language": language  # Metadata
            }

            samples_by_lang[language].append(sample)

            # Data augmentation
            if augment:
                variations = generate_variations(instruction, topic, language)
                for variation in variations:
                    samples_by_lang[language].append({
                        "instruction": variation,
                        "input": input_text,
                        "output": text,
                        "language": language
                    })

    return samples_by_lang


def generate_variations(original: str, topic: str, language: str) -> List[str]:
    """Generate question variations based on language"""
    variations = []
    
    if language == 'ko':
        # Korean variations
        if "설명해주세요" in original:
            variations.append(original.replace("설명해주세요", "설명해줘"))
            variations.append(original.replace("설명해주세요", "간단히 설명해주세요"))
        elif "무엇인가요" in original:
            variations.append(original.replace("무엇인가요", "뭔가요"))
    
    else:  # English
        if "Explain" in original:
            variations.append(original.replace("Explain", "Can you explain"))
            variations.append(original.replace("Explain", "Please explain"))
        elif "What is" in original:
            variations.append(original.replace("What is", "Could you describe"))
    
    return variations[:2]


def filter_quality(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Quality filtering"""
    filtered = []

    for sample in samples:
        instruction = sample['instruction']
        output = sample['output']

        if len(instruction) < 10 or len(output) < 30:
            continue
        if len(output) > 4000:
            continue

        filtered.append(sample)

    return filtered


def deduplicate_samples(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicates"""
    unique = []
    seen = set()

    for sample in samples:
        key = (sample['instruction'], sample['output'][:100])
        if key not in seen:
            seen.add(key)
            unique.append(sample)

    return unique


def save_jsonl(samples: List[Dict[str, str]], output_path: Path, strip_metadata: bool = True):
    """Save in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Remove language metadata if requested
            if strip_metadata and 'language' in sample:
                sample = {k: v for k, v in sample.items() if k != 'language'}
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f" Saved: {output_path} ({len(samples)} samples)")


def print_statistics(samples_by_lang: Dict[str, List]):
    """Print dataset statistics by language"""
    print("\n Dataset Statistics by Language:")
    
    total = 0
    for lang, samples in samples_by_lang.items():
        if samples:
            count = len(samples)
            total += count
            avg_inst = sum(len(s['instruction']) for s in samples) / count
            avg_out = sum(len(s['output']) for s in samples) / count
            
            lang_name = {'ko': 'Korean', 'en': 'English', 'mixed': 'Mixed'}[lang]
            print(f"\n  {lang_name} ({lang.upper()}):")
            print(f"    Samples: {count}")
            print(f"    Avg question: {avg_inst:.1f} chars")
            print(f"    Avg answer: {avg_out:.1f} chars")
    
    print(f"\n  Total: {total} samples")


def main():
    parser = argparse.ArgumentParser(description="Extract training data with auto language detection")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="Document ID list")
    parser.add_argument("--output", type=str, default="data/training_data_auto.jsonl", help="Output file")
    parser.add_argument("--limit", type=int, default=500, help="Max chunks per document")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--separate-langs", action="store_true", help="Save separate files per language")

    args = parser.parse_args()

    print("="*80)
    print(" Milvus Training Data Extraction (Auto Language Detection)")
    print("="*80)
    print(f"Documents: {len(args.doc_ids)}")
    print(f"Max chunks per doc: {args.limit}")
    print(f"Data augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    print(f"Separate by language: {'Yes' if args.separate_langs else 'No'}")
    print("="*80)

    # Connect to Milvus
    connect_milvus()

    # Extract and generate QA samples
    all_samples_by_lang = {'ko': [], 'en': [], 'mixed': []}

    for doc_id in args.doc_ids:
        print(f"\nProcessing: {doc_id}")
        chunks = extract_chunks_by_doc(doc_id, args.limit)

        if not chunks:
            continue

        # Generate QA samples with language detection
        samples_by_lang = generate_qa_samples(chunks, doc_id, augment=not args.no_augment)
        
        for lang, samples in samples_by_lang.items():
            all_samples_by_lang[lang].extend(samples)
            if samples:
                print(f"   {lang.upper()}: {len(samples)} samples")

    # Quality filtering and deduplication per language
    for lang in all_samples_by_lang.keys():
        samples = all_samples_by_lang[lang]
        if not samples:
            continue
        
        print(f"\n Processing {lang.upper()} samples...")
        filtered = filter_quality(samples)
        unique = deduplicate_samples(filtered)
        all_samples_by_lang[lang] = unique
        print(f"  {len(samples)} → {len(filtered)} → {len(unique)}")

    # Save
    output_path = Path(args.output)
    
    if args.separate_langs:
        # Save separate files per language
        for lang, samples in all_samples_by_lang.items():
            if samples:
                lang_output = output_path.parent / f"{output_path.stem}_{lang}{output_path.suffix}"
                save_jsonl(samples, lang_output)
    else:
        # Save all in one file
        all_samples = []
        for samples in all_samples_by_lang.values():
            all_samples.extend(samples)
        save_jsonl(all_samples, output_path)

    # Statistics
    print_statistics(all_samples_by_lang)

    # Sample output
    print("\n Sample outputs per language:")
    print("-"*80)
    for lang, samples in all_samples_by_lang.items():
        if samples:
            lang_name = {'ko': 'Korean', 'en': 'English', 'mixed': 'Mixed'}[lang]
            print(f"\n[{lang_name} Sample]")
            sample = samples[0]
            print(f"Q: {sample['instruction']}")
            if sample.get('input'):
                print(f"Input: {sample['input']}")
            print(f"A: {sample['output'][:150]}...")
            print("-"*80)

    print("\n Completed!")


if __name__ == "__main__":
    main()