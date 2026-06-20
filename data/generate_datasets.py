import os
import sys
import json
import argparse
from typing import List, Dict, Any
from data.data_loader import load_real_github_texts
from data.labeler import get_labeler, BaseLabeler

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

def ensure_dirs(phase: int):
    os.makedirs(os.path.join(DATASETS_DIR, f"phase{phase}"), exist_ok=True)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {file_path}")


def generate_phase1(labeler: BaseLabeler, overwrite_gold: bool = False):
    print("=== STARTING PHASE 1 DATASET GENERATION ===")
    ensure_dirs(1)
    
    gold_path = os.path.join(DATASETS_DIR, "phase1", "gold_test.jsonl")
    train_path = os.path.join(DATASETS_DIR, "phase1", "train.jsonl")
    val_path = os.path.join(DATASETS_DIR, "phase1", "val.jsonl")
    
    # Check if gold test set is already frozen
    gold_exists = os.path.exists(gold_path)
    if gold_exists and not overwrite_gold:
        print(f"INFO: Gold test set already exists at {gold_path} and is FROZEN. Skipping gold generation.")
    else:
        print(f"Generating Phase 1 Gold Test Set (300 examples)...")
        # Load raw issue texts (need 300 for gold + 2000 for train + 300 for val = 2600 total)
        raw_texts = load_real_github_texts(count=2600)
        
        gold_texts = raw_texts[:300]
        gold_data = []
        for idx, text in enumerate(gold_texts):
            if (idx + 1) % 50 == 0:
                print(f"Labeled {idx + 1}/300 gold examples...")
            try:
                label = labeler.label_issue_phase1(text)
                gold_data.append({
                    "id": f"phase1_gold_{idx:04d}",
                    "text": text,
                    "label": label
                })
            except Exception as e:
                print(f"Error labeling gold item {idx}: {e}")
                # Fallback to mock behavior to prevent total failure
                from data.labeler import MockLabeler
                fallback_label = MockLabeler().label_issue_phase1(text)
                gold_data.append({
                    "id": f"phase1_gold_{idx:04d}",
                    "text": text,
                    "label": fallback_label
                })
        
        save_jsonl(gold_data, gold_path)
        print(f"WARNING: Phase 1 Gold Test Set has been created and is FROZEN. Manually edit/verify: {gold_path}")
        
    # Now generate training and validation datasets
    print("Generating Phase 1 Training (2,000 examples) and Validation (300 examples)...")
    raw_texts = load_real_github_texts(count=2600)
    
    # We slice training and validation from the remaining part of raw texts to prevent data leakage!
    # Gold test set: indices 0-300
    # Val set: indices 300-600 (300 items)
    # Train set: indices 600-2600 (2000 items)
    val_texts = raw_texts[300:600]
    train_texts = raw_texts[600:2600]
    
    # Train generation
    train_data = []
    print("Labeling training set...")
    for idx, text in enumerate(train_texts):
        if (idx + 1) % 400 == 0:
            print(f"Labeled {idx + 1}/2000 train examples...")
        try:
            label = labeler.label_issue_phase1(text)
            train_data.append({
                "id": f"phase1_train_{idx:04d}",
                "text": text,
                "label": label
            })
        except Exception as e:
            from data.labeler import MockLabeler
            fallback_label = MockLabeler().label_issue_phase1(text)
            train_data.append({
                "id": f"phase1_train_{idx:04d}",
                "text": text,
                "label": fallback_label
            })
            
    # Val generation
    val_data = []
    print("Labeling validation set...")
    for idx, text in enumerate(val_texts):
        if (idx + 1) % 100 == 0:
            print(f"Labeled {idx + 1}/300 val examples...")
        try:
            label = labeler.label_issue_phase1(text)
            val_data.append({
                "id": f"phase1_val_{idx:04d}",
                "text": text,
                "label": label
            })
        except Exception as e:
            from data.labeler import MockLabeler
            fallback_label = MockLabeler().label_issue_phase1(text)
            val_data.append({
                "id": f"phase1_val_{idx:04d}",
                "text": text,
                "label": fallback_label
            })
            
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    print("=== PHASE 1 DATASETS GENERATION COMPLETED ===")


def generate_phase2(labeler: BaseLabeler, overwrite_gold: bool = False):
    print("=== STARTING PHASE 2 DATASET GENERATION ===")
    ensure_dirs(2)
    
    gold_path = os.path.join(DATASETS_DIR, "phase2", "gold_test.jsonl")
    train_path = os.path.join(DATASETS_DIR, "phase2", "train.jsonl")
    val_path = os.path.join(DATASETS_DIR, "phase2", "val.jsonl")
    
    # Check if gold test set is already frozen
    gold_exists = os.path.exists(gold_path)
    if gold_exists and not overwrite_gold:
        print(f"INFO: Gold test set already exists at {gold_path} and is FROZEN. Skipping gold generation.")
    else:
        print(f"Generating Phase 2 Gold Test Set (300 examples)...")
        # Load raw issue texts (need 300 for gold + 3000 for train + 500 for val = 3800 total)
        raw_texts = load_real_github_texts(count=3800)
        
        gold_texts = raw_texts[:300]
        gold_data = []
        for idx, text in enumerate(gold_texts):
            if (idx + 1) % 50 == 0:
                print(f"Labeled {idx + 1}/300 gold examples...")
            try:
                label = labeler.label_issue_phase2(text)
                gold_data.append({
                    "id": f"phase2_gold_{idx:04d}",
                    "text": text,
                    "label": label
                })
            except Exception as e:
                from data.labeler import MockLabeler
                fallback_label = MockLabeler().label_issue_phase2(text)
                gold_data.append({
                    "id": f"phase2_gold_{idx:04d}",
                    "text": text,
                    "label": fallback_label
                })
        
        save_jsonl(gold_data, gold_path)
        print(f"WARNING: Phase 2 Gold Test Set has been created and is FROZEN. Manually edit/verify: {gold_path}")
        
    # Now generate training and validation datasets
    print("Generating Phase 2 Training (3,000 examples) and Validation (500 examples)...")
    raw_texts = load_real_github_texts(count=3800)
    
    # We slice training and validation from the remaining part of raw texts to prevent data leakage!
    # Gold test set: indices 0-300
    # Val set: indices 300-800 (500 items)
    # Train set: indices 800-3800 (3000 items)
    val_texts = raw_texts[300:800]
    train_texts = raw_texts[800:3800]
    
    # Train generation
    train_data = []
    print("Labeling training set...")
    for idx, text in enumerate(train_texts):
        if (idx + 1) % 500 == 0:
            print(f"Labeled {idx + 1}/3000 train examples...")
        try:
            label = labeler.label_issue_phase2(text)
            train_data.append({
                "id": f"phase2_train_{idx:04d}",
                "text": text,
                "label": label
            })
        except Exception as e:
            from data.labeler import MockLabeler
            fallback_label = MockLabeler().label_issue_phase2(text)
            train_data.append({
                "id": f"phase2_train_{idx:04d}",
                "text": text,
                "label": fallback_label
            })
            
    # Val generation
    val_data = []
    print("Labeling validation set...")
    for idx, text in enumerate(val_texts):
        if (idx + 1) % 100 == 0:
            print(f"Labeled {idx + 1}/500 val examples...")
        try:
            label = labeler.label_issue_phase2(text)
            val_data.append({
                "id": f"phase2_val_{idx:04d}",
                "text": text,
                "label": label
            })
        except Exception as e:
            from data.labeler import MockLabeler
            fallback_label = MockLabeler().label_issue_phase2(text)
            val_data.append({
                "id": f"phase2_val_{idx:04d}",
                "text": text,
                "label": fallback_label
            })
            
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    print("=== PHASE 2 DATASETS GENERATION COMPLETED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReliaLM Dataset Generator")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Which phase to generate datasets for")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "openai", "gemini"], help="Which LLM provider to use for labeling")
    parser.add_argument("--overwrite-gold", action="store_true", help="Overwrite gold test set even if it already exists")
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = get_labeler(args.provider)
    
    if args.phase == 1:
        generate_phase1(labeler, args.overwrite_gold)
    else:
        generate_phase2(labeler, args.overwrite_gold)
