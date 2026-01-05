#!/usr/bin/env python3
# finetune/extract_structured_compliance.py
"""
60% Structured Extraction + 40% Compliance Mapping
Optimized for IAEA official letters (2-20 pages)

Data Distribution:
- 60%: Structured Extraction (JSON, dates, numbers, entities)
- 40%: Compliance Mapping (legal provisions, evidence, actions)

Output Format: JSONL with 4 task types
"""
import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path
from pymilvus import connections, Collection
from datetime import datetime

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks_v2")

# Task distribution (60:40)
TASK_WEIGHTS = {
    "structured_extraction": 0.35,  # JSON extraction
    "extractive_qa": 0.25,          # Short precise QA
    "compliance_mapping": 0.30,     # Legal mapping
    "official_summary": 0.10        # Official summary
}


# ==================== Task Type 1: Structured Extraction ====================
STRUCTURED_SCHEMA = """Extract the following information in JSON format:
{
  "doc_type": "official_letter|measurement_report|schedule_notice|technical_report",
  "entities": {
    "sender_org": "",
    "receiver_org": "",
    "case_id": "",
    "country": "",
    "facility": ""
  },
  "events": [
    {"title": "", "date": "", "time": "", "location": "", "timezone": ""}
  ],
  "nuclear_materials": [
    {"material": "", "quantity": "", "unit": "", "measurement_date": "", "facility": ""}
  ],
  "deadlines": [
    {"item": "", "due_date": "", "basis_text_quote": ""}
  ],
  "key_numbers": [
    {"metric": "", "value": "", "unit": "", "context": ""}
  ]
}"""

def generate_structured_extraction(chunk: Dict[str, Any], doc_id: str) -> List[Dict]:
    """Generate structured JSON extraction tasks"""
    text = chunk.get('chunk', '').strip()
    if len(text) < 100:  # Too short for structured data
        return []
    
    samples = []
    section = chunk.get('section', '')
    page = chunk.get('page', 0)
    
    # Pattern 1: Full document structure
    if any(kw in text.lower() for kw in ['letter', 'report', 'notification', 'inspection']):
        samples.append({
            "instruction": "Extract all structured information from this document in JSON format.",
            "input": f"Document: {doc_id}\n\nSchema:\n{STRUCTURED_SCHEMA}",
            "output": f"Based on the provided text, here is the structured extraction:\n\n{text[:800]}\n\nNote: Extract exact quotes for all dates, numbers, and entities. Use null for missing fields.",
            "task_type": "structured_extraction",
            "output_format": "json"
        })
    
    # Pattern 2: Entity extraction only
    if re.search(r'(from|to|sender|receiver|organization)', text, re.I):
        samples.append({
            "instruction": "Extract sender, receiver, and case ID from this text.",
            "input": f"Document: {doc_id}, Page: {page}",
            "output": f"Entities found in text:\n{text[:600]}\n\n**Important**: Copy exact organization names and case IDs as they appear.",
            "task_type": "structured_extraction",
            "output_format": "json"
        })
    
    # Pattern 3: Dates and deadlines
    date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', text)
    if date_patterns:
        samples.append({
            "instruction": "Extract all dates and deadlines with their context.",
            "input": f"Document: {doc_id}",
            "output": f"Dates found (copy exact format):\n{text[:700]}\n\nDeadlines must include:\n- Exact date\n- What is due\n- Quote from text as basis",
            "task_type": "structured_extraction",
            "output_format": "json"
        })
    
    # Pattern 4: Nuclear materials and quantities
    if any(kw in text.lower() for kw in ['uranium', 'plutonium', 'nuclear material', 'kg', 'gram', 'mtu', 'inventory']):
        samples.append({
            "instruction": "Extract nuclear material information including quantities and units.",
            "input": f"Document: {doc_id}, Section: {section}",
            "output": f"Nuclear materials (exact numbers only):\n{text[:650]}\n\n**Critical**: Copy numbers and units exactly as written. Do not round or estimate.",
            "task_type": "structured_extraction",
            "output_format": "json"
        })
    
    # Pattern 5: Key metrics extraction
    number_patterns = re.findall(r'\b\d+\.?\d*\s*(?:kg|g|MW|%|ppm|rem|Sv|Bq)\b', text)
    if number_patterns:
        samples.append({
            "instruction": "Extract all numerical measurements with units and context.",
            "input": f"Document: {doc_id}",
            "output": f"Key numbers from text:\n{text[:600]}\n\nFormat: [{{'metric': '', 'value': '', 'unit': '', 'context': ''}}]",
            "task_type": "structured_extraction",
            "output_format": "json"
        })
    
    return samples


# ==================== Task Type 2: Extractive QA ====================
def generate_extractive_qa(chunk: Dict[str, Any], doc_id: str) -> List[Dict]:
    """Generate short, precise QA pairs with evidence"""
    text = chunk.get('chunk', '').strip()
    if len(text) < 50:
        return []
    
    samples = []
    section = chunk.get('section', '')
    
    # Pattern 1: Who/When/Where questions
    qa_patterns = [
        (r'(sent|submitted|reported) (?:on|at) ([A-Z][a-z]+ \d+, \d{4})', "When was this document sent?"),
        (r'(from|by) ([A-Z][A-Za-z\s]+(?:Organization|Agency|Authority))', "Who sent this document?"),
        (r'(to) ([A-Z][A-Za-z\s]+(?:Organization|Agency|Authority))', "Who is the recipient?"),
        (r'case (?:number|ID|reference):?\s*([A-Z0-9/-]+)', "What is the case ID?"),
        (r'facility:?\s*([A-Z][A-Za-z\s]+)', "Which facility is mentioned?"),
    ]
    
    for pattern, question in qa_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            answer = match.group(1) if match.groups() else "Not found"
            samples.append({
                "instruction": question,
                "input": f"Document: {doc_id}",
                "output": f"{answer}\n\nEvidence: \"{text[:200]}...\"",
                "task_type": "extractive_qa",
                "output_format": "text"
            })
    
    # Pattern 2: Quantity questions
    if any(kw in text.lower() for kw in ['quantity', 'amount', 'measurement', 'inventory']):
        samples.append({
            "instruction": "What quantity of nuclear material is mentioned?",
            "input": f"Document: {doc_id}, Section: {section}",
            "output": f"Based on the text: {text[:400]}\n\n**Answer**: [Quote exact number and unit]\n\nIf not found: \"No quantity specified in this section.\"",
            "task_type": "extractive_qa",
            "output_format": "text"
        })
    
    # Pattern 3: Deadline questions
    if any(kw in text.lower() for kw in ['deadline', 'due date', 'by', 'before', 'no later than']):
        samples.append({
            "instruction": "What is the deadline mentioned?",
            "input": f"Document: {doc_id}",
            "output": f"From text: {text[:350]}\n\n**Deadline**: [Exact date]\n**For**: [What is due]\n\nIf no deadline: \"No specific deadline found.\"",
            "task_type": "extractive_qa",
            "output_format": "text"
        })
    
    # Pattern 4: "Not found" training (critical for hallucination prevention)
    samples.append({
        "instruction": "What is the shipment tracking number?",
        "input": f"Document: {doc_id}",
        "output": "Not found in this document.\n\nNote: I searched the provided text but did not find any shipment tracking number. Only provide information that is explicitly stated.",
        "task_type": "extractive_qa",
        "output_format": "text"
    })
    
    return samples


# ==================== Task Type 3: Compliance Mapping ====================
COMPLIANCE_TEMPLATE = """[Relevant Provisions]
- Provision: {provision_name}
- Why it applies: {reason}
- Evidence from letter: "{quote}"

[Required Actions]
- Action: {action_description}
- Owner: {responsible_party}
- Deadline: {due_date}
- Evidence: "{evidence_quote}"

[Compliance Status]
- Current status: {status}
- Gaps (if any): {gaps}"""

def generate_compliance_mapping(chunk: Dict[str, Any], doc_id: str) -> List[Dict]:
    """Generate legal compliance mapping tasks"""
    text = chunk.get('chunk', '').strip()
    if len(text) < 80:
        return []
    
    samples = []
    
    # Pattern 1: Legal provision references
    provision_patterns = [
        r'Article\s+\d+',
        r'Section\s+\d+\.\d+',
        r'Requirement\s+\d+',
        r'Regulation\s+\([A-Z]+\)\s+No\.\s+\d+',
        r'INFCIRC/\d+',
        r'제\s*\d+\s*조',  # Korean law articles
    ]
    
    for pattern in provision_patterns:
        if re.search(pattern, text, re.I):
            samples.append({
                "instruction": "Map this text to relevant legal provisions and required actions.",
                "input": f"Document: {doc_id}\n\nText:\n{text[:600]}\n\nFormat:\n{COMPLIANCE_TEMPLATE}",
                "output": f"**[Relevant Provisions]**\n- Provision: [Extract from text]\n- Why it applies: [Explain based on content]\n- Evidence: \"[Direct quote]\"\n\n**[Required Actions]**\n- Action: [What must be done]\n- Owner: [Who is responsible]\n- Deadline: [When, if specified]\n- Evidence: \"[Quote showing requirement]\"\n\n**Critical**: All quotes must be exact. If no deadline, state \"Not specified\".",
                "task_type": "compliance_mapping",
                "output_format": "structured_text"
            })
            break  # One per chunk to avoid redundancy
    
    # Pattern 2: Obligations and requirements
    if any(kw in text.lower() for kw in ['shall', 'must', 'required', 'obligation', 'mandatory']):
        samples.append({
            "instruction": "Identify all compliance obligations in this text.",
            "input": f"Document: {doc_id}",
            "output": f"Compliance analysis:\n\n{text[:500]}\n\n**Obligations**:\n1. [List each 'shall/must' statement]\n2. [Include legal basis if referenced]\n\n**Evidence**: Quote exact text for each obligation.\n\n**Note**: If assumption needed, clearly state: \"Assumption: ...\"",
            "task_type": "compliance_mapping",
            "output_format": "structured_text"
        })
    
    # Pattern 3: Safeguards-specific compliance
    if any(kw in text.lower() for kw in ['safeguards', 'verification', 'inspection', 'declaration', 'inventory']):
        samples.append({
            "instruction": "What are the safeguards compliance requirements?",
            "input": f"Document: {doc_id}\n\nSafeguards context:\n{text[:550]}",
            "output": "**Safeguards Requirements**:\n- Type: [PIV/DIV/Inspection/etc.]\n- Basis: [INFCIRC/Agreement reference]\n- Action required: [Specific task]\n- Deadline: [If stated]\n- Evidence: \"[Quote]\"\n\n**Verification**: All information must be directly from the text.",
            "task_type": "compliance_mapping",
            "output_format": "structured_text"
        })
    
    # Pattern 4: Gap analysis template
    samples.append({
        "instruction": "Analyze compliance status and identify any gaps.",
        "input": f"Document: {doc_id}",
        "output": f"Based on: {text[:400]}\n\n**Compliance Status**:\n- Requirements met: [List]\n- Outstanding items: [List]\n- Gaps: [Identify missing info]\n\n**Evidence for each**: Quote supporting text.\n\n**If information is incomplete**: Clearly state \"Insufficient information to determine [aspect]\".",
        "task_type": "compliance_mapping",
        "output_format": "structured_text"
    })
    
    return samples


# ==================== Task Type 4: Official Summary ====================
def generate_official_summary(chunk: Dict[str, Any], doc_id: str) -> List[Dict]:
    """Generate business-style summaries for official letters"""
    text = chunk.get('chunk', '').strip()
    if len(text) < 150:  # Need substantial text for summary
        return []
    
    samples = []
    
    # Pattern 1: Full document summary
    samples.append({
        "instruction": "Provide an official summary of this document using the standard format.",
        "input": f"Document: {doc_id}\n\nFull text:\n{text[:800]}",
        "output": """**When**: [Date of document/event]
**Where**: [Location/Facility]
**Who**: [Sender → Receiver]
**What**: [Key content in 1-2 sentences]
**Action Required**: [What must be done]
**Deadline**: [When, if applicable]
**Nuclear Material**: [Type, quantity if mentioned]

**Key Numbers** (copy exactly):
- [Metric 1]: [Value] [Unit]
- [Metric 2]: [Value] [Unit]

**Note**: All dates, numbers, and names are copied verbatim from the original text.""",
        "task_type": "official_summary",
        "output_format": "structured_text"
    })
    
    # Pattern 2: Executive summary for reports
    if any(kw in text.lower() for kw in ['report', 'findings', 'results', 'inspection']):
        samples.append({
            "instruction": "Create an executive summary for this report.",
            "input": f"Document: {doc_id}",
            "output": f"Report context: {text[:500]}\n\n**Executive Summary**:\n- **Purpose**: [Why this report]\n- **Scope**: [What was covered]\n- **Key Findings**: [Main results]\n- **Nuclear Materials**: [If applicable]\n- **Recommendations**: [If any]\n\n**Numerical Data**: Copy all numbers with units exactly as written.",
            "task_type": "official_summary",
            "output_format": "structured_text"
        })
    
    return samples


# ==================== Main Extraction Logic ====================
def connect_milvus():
    """Connect to Milvus"""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f" Connected to Milvus: {MILVUS_HOST}:{MILVUS_PORT}")


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


def generate_all_tasks(chunks: List[Dict], doc_id: str) -> Dict[str, List[Dict]]:
    """Generate all 4 task types with target distribution"""
    all_samples = {
        "structured_extraction": [],
        "extractive_qa": [],
        "compliance_mapping": [],
        "official_summary": []
    }
    
    for chunk in chunks:
        # Generate each task type
        all_samples["structured_extraction"].extend(
            generate_structured_extraction(chunk, doc_id)
        )
        all_samples["extractive_qa"].extend(
            generate_extractive_qa(chunk, doc_id)
        )
        all_samples["compliance_mapping"].extend(
            generate_compliance_mapping(chunk, doc_id)
        )
        all_samples["official_summary"].extend(
            generate_official_summary(chunk, doc_id)
        )
    
    return all_samples


def balance_samples(samples_by_type: Dict[str, List[Dict]], total_target: int) -> List[Dict]:
    """Balance samples according to target weights (60:40 split)"""
    balanced = []
    
    for task_type, weight in TASK_WEIGHTS.items():
        task_samples = samples_by_type.get(task_type, [])
        target_count = int(total_target * weight)
        
        if len(task_samples) >= target_count:
            # Randomly sample if too many
            import random
            balanced.extend(random.sample(task_samples, target_count))
        else:
            # Use all if too few
            balanced.extend(task_samples)
        
        print(f"  {task_type}: {len(task_samples)} → {min(len(task_samples), target_count)} samples")
    
    return balanced


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Remove task_type metadata before saving
            clean = {k: v for k, v in sample.items() if k != 'task_type'}
            f.write(json.dumps(clean, ensure_ascii=False) + '\n')
    
    print(f"\n Saved: {output_path} ({len(samples)} samples)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="60/40 Structured Extraction + Compliance Mapping")
    parser.add_argument("--doc-ids", nargs="+", required=True, help="Document IDs")
    parser.add_argument("--output", default="data/training_structured_compliance.jsonl", help="Output file")
    parser.add_argument("--total-samples", type=int, default=5000, help="Total target samples")
    
    args = parser.parse_args()
    
    print("="*80)
    print(" Structured Extraction + Compliance Mapping Data Generation")
    print("="*80)
    print(f"Distribution: 60% Extraction (JSON/QA) + 40% Compliance")
    print(f"  - Structured Extraction: 35%")
    print(f"  - Extractive QA: 25%")
    print(f"  - Compliance Mapping: 30%")
    print(f"  - Official Summary: 10%")
    print(f"Total target: {args.total_samples} samples")
    print("="*80)
    
    connect_milvus()
    
    # Extract and generate
    all_samples_by_type = {
        "structured_extraction": [],
        "extractive_qa": [],
        "compliance_mapping": [],
        "official_summary": []
    }
    
    for doc_id in args.doc_ids:
        print(f"\nProcessing: {doc_id}")
        chunks = extract_chunks_by_doc(doc_id)
        
        if not chunks:
            continue
        
        # Generate all task types
        doc_samples = generate_all_tasks(chunks, doc_id)
        
        # Accumulate
        for task_type, samples in doc_samples.items():
            all_samples_by_type[task_type].extend(samples)
            print(f"  {task_type}: +{len(samples)} samples")
    
    # Balance according to weights
    print("\n Balancing samples...")
    balanced_samples = balance_samples(all_samples_by_type, args.total_samples)
    
    # Shuffle
    import random
    random.shuffle(balanced_samples)
    
    # Save
    save_jsonl(balanced_samples, Path(args.output))
    
    # Statistics
    print("\n Final Distribution:")
    task_counts = {}
    for sample in balanced_samples:
        task = sample.get('task_type', 'unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        ratio = count / len(balanced_samples) * 100 if balanced_samples else 0
        print(f"  {task}: {count} ({ratio:.1f}%)")
    
    print(f"\nTotal: {len(balanced_samples)} samples")
    print("\n Extraction completed!")


if __name__ == "__main__":
    main()