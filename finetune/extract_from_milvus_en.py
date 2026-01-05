#!/usr/bin/env python3
# finetune/extract_from_milvus_en.py
"""
Extract training data from Milvus for English documents (IAEA guidelines, technical manuals)

Usage:
    python finetune/extract_from_milvus_en.py --doc-ids DOC1 DOC2 --output data/training_data_en.jsonl

Features:
    - IAEA Safety Standards specific patterns
    - Technical manual QA generation
    - Nuclear engineering terminology focus
    - Data augmentation with paraphrasing
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

# English QA generation patterns
QA_PATTERNS_EN = {
    "iaea_safety": {  # IAEA Safety Standards
        "keywords": ["iaea", "safety standard", "requirement", "shall", "should", "must"],
        "templates": [
            "What are the IAEA requirements for {topic}?",
            "Explain the safety standards for {topic}.",
            "What does IAEA say about {topic}?",
            "Describe the requirements for {topic} according to IAEA."
        ]
    },
    "technical": {  # Technical descriptions
        "keywords": ["system", "component", "design", "function", "operation", "consists of"],
        "templates": [
            "Describe the {topic}.",
            "What is the function of {topic}?",
            "How does {topic} work?",
            "Explain the design of {topic}."
        ]
    },
    "procedure": {  # Procedures and methods
        "keywords": ["procedure", "method", "step", "process", "perform", "conduct"],
        "templates": [
            "What is the procedure for {topic}?",
            "How to perform {topic}?",
            "Describe the steps for {topic}.",
            "Explain the method for {topic}."
        ]
    },
    "definition": {  # Definitions and terminology
        "keywords": ["means", "refers to", "defined as", "is", "definition"],
        "templates": [
            "What is {topic}?",
            "Define {topic}.",
            "What does {topic} mean?",
            "Explain the concept of {topic}."
        ]
    },
    "safety_principle": {  # Safety principles
        "keywords": ["defence in depth", "redundancy", "diversity", "independence", "fail-safe"],
        "templates": [
            "Explain the principle of {topic}.",
            "What is the {topic} principle?",
            "How is {topic} applied in nuclear safety?",
            "Describe the concept of {topic}."
        ]
    },
    "regulatory": {  # Regulatory and compliance
        "keywords": ["regulation", "compliance", "criteria", "limit", "threshold", "acceptable"],
        "templates": [
            "What are the regulatory requirements for {topic}?",
            "What are the acceptance criteria for {topic}?",
            "Explain the limits for {topic}.",
            "What are the compliance requirements for {topic}?"
        ]
    },
    "accident": {  # Accident scenarios and emergency
        "keywords": ["accident", "emergency", "response", "mitigation", "dba", "beyond design basis"],
        "templates": [
            "What is the response for {topic}?",
            "How to mitigate {topic}?",
            "Describe the emergency procedures for {topic}.",
            "What are the safety measures for {topic}?"
        ]
    }
}


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

        # Query by doc_id
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


def classify_chunk_type(chunk: Dict[str, Any]) -> str:
    """Classify document type based on content"""
    text = chunk.get('chunk', '').lower()
    section = chunk.get('section', '').lower()

    # IAEA Safety Standards
    if any(kw in text for kw in QA_PATTERNS_EN["iaea_safety"]["keywords"]):
        if "iaea" in text or "safety standard" in text:
            return "iaea_safety"

    # Accident/Emergency
    if any(kw in text for kw in QA_PATTERNS_EN["accident"]["keywords"]):
        return "accident"

    # Regulatory
    if any(kw in text for kw in QA_PATTERNS_EN["regulatory"]["keywords"]):
        return "regulatory"

    # Safety principles
    if any(kw in text for kw in QA_PATTERNS_EN["safety_principle"]["keywords"]):
        return "safety_principle"

    # Procedures
    if any(kw in text for kw in QA_PATTERNS_EN["procedure"]["keywords"]):
        return "procedure"

    # Definitions
    if any(kw in text for kw in QA_PATTERNS_EN["definition"]["keywords"]):
        return "definition"

    # Technical
    if any(kw in text for kw in QA_PATTERNS_EN["technical"]["keywords"]):
        return "technical"

    return "general"


def extract_topic(chunk: Dict[str, Any], chunk_type: str) -> str:
    """Extract key topic from chunk"""
    text = chunk.get('chunk', '')
    section = chunk.get('section', '')

    # Use section title if available
    if section and section not in ["Unknown", "META"]:
        return section.strip()

    # IAEA documents: extract from headings
    if chunk_type == "iaea_safety":
        # Pattern: "3.2 Defence in Depth" or "Requirement 5: ..."
        match = re.search(r'(?:Requirement|Guideline|Section)\s+\d+[:\.]?\s*([A-Z][a-zA-Z\s]+)', text)
        if match:
            return match.group(1).strip()

    # Technical: extract system/component names
    if chunk_type == "technical":
        # Pattern: "The reactor coolant system" or "Emergency core cooling system (ECCS)"
        match = re.search(r'(?:The\s+)?([A-Z][a-z]+(?:\s+[a-z]+){1,4}(?:\s+system|\s+component)?)', text)
        if match:
            return match.group(1).strip()

    # Definition: extract term being defined
    if chunk_type == "definition":
        # Pattern: "Defence in depth means..." or "ALARA is defined as..."
        match = re.search(r'^([A-Z][A-Za-z\s]+?)\s+(?:means|is defined as|refers to)', text)
        if match:
            return match.group(1).strip()

    # Extract from first sentence (simple heuristic)
    first_sentence = text.split('.')[0].split('\n')[0]
    if len(first_sentence) > 10 and len(first_sentence) < 100:
        # Remove common sentence starters
        cleaned = re.sub(r'^(The|A|An)\s+', '', first_sentence)
        return cleaned.strip()

    return "this topic"


def generate_qa_samples(
    chunks: List[Dict[str, Any]],
    doc_id: str,
    augment: bool = True
) -> List[Dict[str, str]]:
    """Generate QA samples from chunks"""
    samples = []

    for chunk in chunks:
        text = chunk.get('chunk', '').strip()
        if not text or len(text) < 50:  # Skip very short chunks
            continue

        # Remove meta lines
        text = re.sub(r'^META:.*?\n', '', text)

        chunk_type = classify_chunk_type(chunk)
        topic = extract_topic(chunk, chunk_type)
        section = chunk.get('section', '')
        page = chunk.get('page', 0)

        # Select templates based on chunk type
        if chunk_type in QA_PATTERNS_EN:
            templates = QA_PATTERNS_EN[chunk_type]["templates"]
        else:
            templates = [
                "Explain {topic}.",
                "What is {topic}?",
                "Describe {topic}."
            ]

        # Generate questions using templates
        for template in templates[:2]:  # Max 2 per template
            instruction = template.format(topic=topic)

            # Build input field (document context)
            input_parts = []
            if doc_id:
                input_parts.append(f"Document: {doc_id}")
            if section and section not in ["Unknown", "META"]:
                input_parts.append(f"Section: {section}")
            if page:
                input_parts.append(f"Page: {page}")

            input_text = ", ".join(input_parts)

            sample = {
                "instruction": instruction,
                "input": input_text,
                "output": text
            }

            samples.append(sample)

            # Data augmentation
            if augment:
                variations = generate_question_variations(instruction, topic, chunk_type)
                for variation in variations:
                    samples.append({
                        "instruction": variation,
                        "input": input_text,
                        "output": text
                    })

    return samples


def generate_question_variations(original: str, topic: str, chunk_type: str) -> List[str]:
    """Generate question variations for data augmentation"""
    variations = []

    # Formal <-> Informal
    if "Explain" in original:
        variations.append(original.replace("Explain", "Can you explain"))
        variations.append(original.replace("Explain", "Please explain"))
    elif "What is" in original:
        variations.append(original.replace("What is", "Could you describe"))
        variations.append(original.replace("What is", "Tell me about"))
    elif "Describe" in original:
        variations.append(original.replace("Describe", "Please describe"))
        variations.append(original.replace("Describe", "Provide details about"))

    # Question type variations
    if "requirements" in original.lower():
        variations.append(f"What are the key requirements for {topic}?")
        variations.append(f"List the requirements for {topic}.")

    if "procedure" in original.lower():
        variations.append(f"What are the steps for {topic}?")
        variations.append(f"Outline the procedure for {topic}.")

    # Add context-specific variations
    if chunk_type == "iaea_safety":
        variations.append(f"According to IAEA, {original.lower()}")
    elif chunk_type == "safety_principle":
        variations.append(f"Explain the safety concept of {topic}.")

    return variations[:2]  # Max 2 variations


def filter_quality(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Quality filtering"""
    filtered = []

    for sample in samples:
        instruction = sample['instruction']
        output = sample['output']

        # Filter conditions
        if len(instruction) < 10:  # Too short question
            continue
        if len(output) < 30:  # Too short answer
            continue
        if len(output) > 4000:  # Too long (token limit)
            continue
        if instruction.count('?') > 3:  # Too complex
            continue

        # English quality check
        if not re.search(r'[a-zA-Z]', instruction):  # No English characters
            continue
        if not re.search(r'[a-zA-Z]', output):  # No English in output
            continue

        filtered.append(sample)

    return filtered


def deduplicate_samples(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicates"""
    unique = []
    seen = set()

    for sample in samples:
        # Create key from instruction + output prefix
        key = (sample['instruction'], sample['output'][:100])
        if key not in seen:
            seen.add(key)
            unique.append(sample)

    return unique


def save_jsonl(samples: List[Dict[str, str]], output_path: Path):
    """Save in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f" Dataset saved: {output_path}")
    print(f"   Total {len(samples)} samples")


def print_statistics(samples: List[Dict[str, str]]):
    """Print dataset statistics"""
    if not samples:
        return

    inst_lengths = [len(s['instruction']) for s in samples]
    out_lengths = [len(s['output']) for s in samples]

    print("\n Dataset Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Avg question length: {sum(inst_lengths) / len(inst_lengths):.1f} chars")
    print(f"   Avg answer length: {sum(out_lengths) / len(out_lengths):.1f} chars")
    print(f"   Max answer length: {max(out_lengths)} chars")
    print(f"   Min answer length: {min(out_lengths)} chars")

    # Input field usage
    with_input = sum(1 for s in samples if s.get('input'))
    print(f"   With context: {with_input}/{len(samples)} ({with_input/len(samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract training data from Milvus (English documents)")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="Document ID list")
    parser.add_argument("--output", type=str, default="data/training_data_en.jsonl", help="Output file")
    parser.add_argument("--limit", type=int, default=500, help="Max chunks per document")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()

    print("="*80)
    print(" Milvus Training Data Extraction (English Documents)")
    print("="*80)
    print(f"Documents: {len(args.doc_ids)}")
    print(f"Max chunks per doc: {args.limit}")
    print(f"Data augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    print("="*80)

    # Connect to Milvus
    connect_milvus()

    # Extract chunks per document
    all_samples = []

    for doc_id in args.doc_ids:
        print(f"\nProcessing: {doc_id}")
        chunks = extract_chunks_by_doc(doc_id, args.limit)

        if not chunks:
            continue

        # Generate QA samples
        samples = generate_qa_samples(chunks, doc_id, augment=not args.no_augment)
        print(f"   {len(samples)} QA samples generated")

        all_samples.extend(samples)

    print(f"\n Total {len(all_samples)} samples generated (including duplicates)")

    # Quality filtering
    print("\n Quality filtering...")
    filtered = filter_quality(all_samples)
    print(f"   {len(all_samples)} → {len(filtered)} samples (removed: {len(all_samples) - len(filtered)})")

    # Deduplication
    print("\n Removing duplicates...")
    unique = deduplicate_samples(filtered)
    print(f"   {len(filtered)} → {len(unique)} samples (removed: {len(filtered) - len(unique)})")

    # Save
    output_path = Path(args.output)
    save_jsonl(unique, output_path)

    # Statistics
    print_statistics(unique)

    # Sample output
    print("\n Sample 3 examples:")
    print("-"*80)
    for i, sample in enumerate(unique[:3]):
        print(f"\n[Sample {i+1}]")
        print(f"Question: {sample['instruction']}")
        if sample.get('input'):
            print(f"Input: {sample['input']}")
        print(f"Answer: {sample['output'][:200]}...")
        print("-"*80)

    print("\n Completed!")


if __name__ == "__main__":
    main()