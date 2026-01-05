#!/usr/bin/env python3
# finetune/extract_english_first.py
"""
English-first extraction strategy optimized for original IAEA documents

Priorities:
1. English original documents (70%)
2. Korean native regulations (20%)
3. Korean translations (10%)
"""
import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from pymilvus import connections, Collection

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

# Document classification
DOC_CATEGORIES = {
    "english_original": {
        "patterns": ["IAEA", "SSR", "NS-G", "SF-", "GS-G"],
        "priority": 1,
        "target_ratio": 0.70
    },
    "korean_native": {
        "patterns": ["원자력안전법", "방사선안전", "방호및방재"],
        "priority": 2,
        "target_ratio": 0.20
    },
    "korean_translation": {
        "patterns": ["번역", "국문", "-KO"],
        "priority": 3,
        "target_ratio": 0.10
    }
}

# Enhanced English patterns for IAEA documents
IAEA_PATTERNS = {
    "requirement": {
        "trigger": r"Requirement\s+\d+:",
        "templates": [
            "What is Requirement {number} about?",
            "Explain Requirement {number} from IAEA {standard}.",
            "What are the key points of Requirement {number}?"
        ]
    },
    "fundamental_principle": {
        "trigger": r"Fundamental Safety Principle|Safety Principle \d+",
        "templates": [
            "Explain the fundamental safety principle of {topic}.",
            "What is the IAEA safety principle regarding {topic}?",
            "How does IAEA define {topic}?"
        ]
    },
    "defence_in_depth": {
        "trigger": r"defence in depth|defense in depth",
        "templates": [
            "Explain the concept of defence in depth.",
            "What are the levels of defence in depth?",
            "How is defence in depth implemented?"
        ]
    },
    "graded_approach": {
        "trigger": r"graded approach",
        "templates": [
            "What is the graded approach in nuclear safety?",
            "How is the graded approach applied?",
            "Explain the graded approach concept."
        ]
    }
}


def classify_document(doc_id: str) -> Dict[str, Any]:
    """Classify document by category and priority"""
    doc_id_lower = doc_id.lower()
    
    for category, config in DOC_CATEGORIES.items():
        for pattern in config["patterns"]:
            if pattern.lower() in doc_id_lower:
                return {
                    "category": category,
                    "priority": config["priority"],
                    "is_original": category == "english_original",
                    "is_native_korean": category == "korean_native"
                }
    
    # Default: assume English original
    return {
        "category": "english_original",
        "priority": 1,
        "is_original": True,
        "is_native_korean": False
    }


def extract_with_priority(
    doc_ids: List[str],
    total_target: int = 5000,
    output_dir: Path = Path("data")
) -> Dict[str, List[Dict]]:
    """
    Extract samples with priority-based distribution
    
    Returns samples categorized by document type
    """
    # Classify all documents
    classified_docs = {}
    for doc_id in doc_ids:
        classification = classify_document(doc_id)
        category = classification["category"]
        
        if category not in classified_docs:
            classified_docs[category] = []
        
        classified_docs[category].append({
            "doc_id": doc_id,
            "classification": classification
        })
    
    print("\n Document Classification:")
    for category, docs in classified_docs.items():
        print(f"  {category}: {len(docs)} documents")
    
    # Calculate samples per category
    samples_per_category = {}
    for category, config in DOC_CATEGORIES.items():
        target_samples = int(total_target * config["target_ratio"])
        samples_per_category[category] = target_samples
        print(f"\n Target for {category}: {target_samples} samples")
    
    # Extract samples per category
    all_samples = {}
    
    for category, docs in classified_docs.items():
        if category not in samples_per_category:
            continue
        
        target = samples_per_category[category]
        samples_per_doc = target // len(docs) if docs else 0
        
        print(f"\n Processing {category}...")
        print(f"   {len(docs)} documents, ~{samples_per_doc} samples each")
        
        category_samples = []
        
        for doc_info in docs:
            doc_id = doc_info["doc_id"]
            is_original = doc_info["classification"]["is_original"]
            
            # Extract chunks
            chunks = extract_chunks_by_doc(doc_id, limit=samples_per_doc)
            
            if not chunks:
                continue
            
            # Generate QA based on document type
            if is_original:
                # Use English patterns for originals
                samples = generate_iaea_qa(chunks, doc_id)
            else:
                # Use Korean patterns for translations/native
                samples = generate_korean_qa(chunks, doc_id)
            
            category_samples.extend(samples)
            print(f"     {doc_id}: {len(samples)} samples")
        
        all_samples[category] = category_samples
        print(f"    Total {category}: {len(category_samples)} samples")
    
    return all_samples


def extract_chunks_by_doc(doc_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Extract chunks from Milvus"""
    try:
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        results = collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["doc_id", "seq", "page", "section", "chunk"],
            limit=limit
        )

        return results

    except Exception as e:
        print(f"   Failed to extract {doc_id}: {e}")
        return []


def generate_iaea_qa(chunks: List[Dict], doc_id: str) -> List[Dict[str, str]]:
    """Generate QA for IAEA English documents"""
    samples = []
    
    for chunk in chunks:
        text = chunk.get('chunk', '').strip()
        if len(text) < 50:
            continue
        
        section = chunk.get('section', '')
        page = chunk.get('page', 0)
        
        # Pattern 1: Requirement-based
        import re
        req_match = re.search(r'Requirement\s+(\d+)', text)
        if req_match:
            req_num = req_match.group(1)
            samples.append({
                "instruction": f"What is Requirement {req_num} about?",
                "input": f"Document: {doc_id}, Section: {section}",
                "output": text[:600]
            })
            samples.append({
                "instruction": f"Explain Requirement {req_num} from {doc_id}.",
                "input": "",
                "output": text[:550]
            })
        
        # Pattern 2: Defence in Depth
        if "defence in depth" in text.lower() or "defense in depth" in text.lower():
            samples.append({
                "instruction": "Explain the concept of defence in depth.",
                "input": f"Document: {doc_id}",
                "output": text[:600]
            })
            samples.append({
                "instruction": "What are the levels of defence in depth?",
                "input": f"IAEA Standard: {doc_id}",
                "output": text[:550]
            })
        
        # Pattern 3: Safety principles
        if any(kw in text.lower() for kw in ["safety principle", "fundamental", "shall ensure"]):
            topic = section if section else "nuclear safety"
            samples.append({
                "instruction": f"What are the safety principles for {topic}?",
                "input": f"Document: {doc_id}",
                "output": text[:500]
            })
        
        # Pattern 4: Graded approach
        if "graded approach" in text.lower():
            samples.append({
                "instruction": "What is the graded approach in nuclear safety?",
                "input": f"IAEA: {doc_id}",
                "output": text[:550]
            })
        
        # Pattern 5: Technical systems
        if any(kw in text.lower() for kw in ["system", "component", "equipment"]):
            samples.append({
                "instruction": f"Describe the {section}." if section else "Describe the system.",
                "input": f"Document: {doc_id}, Page: {page}",
                "output": text[:500]
            })
        
        # Pattern 6: General (always include)
        if section and section not in ["Unknown", "META"]:
            samples.append({
                "instruction": f"Explain {section}.",
                "input": f"IAEA Document: {doc_id}",
                "output": text[:450]
            })
    
    return samples


def generate_korean_qa(chunks: List[Dict], doc_id: str) -> List[Dict[str, str]]:
    """Generate QA for Korean documents"""
    samples = []
    
    for chunk in chunks:
        text = chunk.get('chunk', '').strip()
        if len(text) < 50:
            continue
        
        section = chunk.get('section', '')
        
        # Korean native patterns
        import re
        
        # Pattern 1: 법 조항
        if re.search(r'제\s*\d+\s*조', text):
            samples.append({
                "instruction": f"{section}의 내용은 무엇인가요?" if section else "이 법 조항의 내용은?",
                "input": f"문서: {doc_id}",
                "output": text[:500]
            })
        
        # Pattern 2: 일반 질문
        if section:
            samples.append({
                "instruction": f"{section}에 대해 설명해주세요.",
                "input": f"문서: {doc_id}",
                "output": text[:450]
            })
    
    return samples


def save_by_category(samples_dict: Dict[str, List], output_dir: Path):
    """Save samples by category"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category, samples in samples_dict.items():
        if not samples:
            continue
        
        output_file = output_dir / f"training_{category}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Remove category metadata
                clean_sample = {k: v for k, v in sample.items() if k != 'category'}
                f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
        
        print(f"\n Saved: {output_file} ({len(samples)} samples)")


def main():
    parser = argparse.ArgumentParser(description="English-first extraction strategy")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="Document ID list")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--total-samples", type=int, default=5000, help="Total target samples")
    parser.add_argument("--combined", action="store_true", help="Also create combined file")
    
    args = parser.parse_args()
    
    print("="*80)
    print(" English-First Training Data Extraction")
    print("="*80)
    print(f"Documents: {len(args.doc_ids)}")
    print(f"Target samples: {args.total_samples}")
    print(f"Distribution: 70% English, 20% Korean Native, 10% Korean Translation")
    print("="*80)
    
    # Connect to Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f" Connected to Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Extract with priority
    output_dir = Path(args.output_dir)
    samples_by_category = extract_with_priority(
        args.doc_ids,
        total_target=args.total_samples,
        output_dir=output_dir
    )
    
    # Save by category
    save_by_category(samples_by_category, output_dir)
    
    # Create combined file if requested
    if args.combined:
        combined_samples = []
        for samples in samples_by_category.values():
            combined_samples.extend(samples)
        
        combined_file = output_dir / "training_combined.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for sample in combined_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n Combined: {combined_file} ({len(combined_samples)} samples)")
    
    # Statistics
    print("\n" + "="*80)
    print(" Final Statistics:")
    print("="*80)
    
    total = 0
    for category, samples in samples_by_category.items():
        count = len(samples)
        total += count
        ratio = (count / args.total_samples * 100) if args.total_samples > 0 else 0
        print(f"{category:25s}: {count:5d} samples ({ratio:5.1f}%)")
    
    print(f"{'TOTAL':25s}: {total:5d} samples")
    print("="*80)
    
    # Sample outputs
    print("\n Sample from each category:\n")
    for category, samples in samples_by_category.items():
        if samples:
            print(f"[{category.upper()}]")
            sample = samples[0]
            print(f"Q: {sample['instruction']}")
            print(f"A: {sample['output'][:100]}...\n")
    
    print(" Extraction completed!")


if __name__ == "__main__":
    main()