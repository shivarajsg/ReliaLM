import json
from typing import Dict, Any, List, Tuple
from evaluation.evaluator import (
    extract_json_string,
    parse_and_validate_phase1,
    parse_and_validate_phase2
)

def classify_failure_phase1(
    raw_output: str, 
    gold_label: Dict[str, Any], 
    input_text: str
) -> str:
    """Classifies a Phase 1 prediction failure.
    Returns one of: 'malformed_json', 'schema_violation', 'wrong_field_value'.
    If prediction is correct (no failure), returns ''.
    """
    is_json, is_schema, parsed, _ = parse_and_validate_phase1(raw_output)
    
    if not is_json:
        return "malformed_json"
        
    if not is_schema or parsed is None:
        return "schema_violation"
        
    # Check if there is any field mismatch
    fields = ["issue_type", "root_cause", "priority", "affected_component"]
    mismatch = False
    for f in fields:
        gold_val = str(gold_label.get(f, "")).strip().lower()
        pred_val = str(parsed.get(f, "")).strip().lower()
        if gold_val != pred_val:
            mismatch = True
            break
            
    if mismatch:
        return "wrong_field_value"
        
    return ""  # No failure


def classify_failure_phase2(
    raw_output: str, 
    gold_label: Dict[str, Any], 
    input_text: str
) -> str:
    """Classifies a Phase 2 prediction failure.
    Returns one of: 'malformed_json', 'schema_violation', 'wrong_tool', 
    'missing_parameter', 'hallucinated_parameter'.
    If prediction is correct, returns ''.
    """
    is_json, is_schema, parsed, _ = parse_and_validate_phase2(raw_output)
    
    if not is_json:
        return "malformed_json"
        
    if not is_schema or parsed is None:
        return "schema_violation"
        
    pred_tool = parsed.get("tool", "")
    pred_params = parsed.get("parameters", {})
    gold_tool = gold_label.get("tool", "")
    gold_params = gold_label.get("parameters", {})
    
    # 1. Wrong Tool
    if pred_tool != gold_tool:
        return "wrong_tool"
        
    # Required parameters for this tool
    required_params_map = {
        "search_repo": ["query"],
        "read_docs": ["doc_path"],
        "create_issue": ["title", "body"],
        "run_tests": ["test_path"],
        "summarize_pr": ["pr_number"]
    }
    
    # Check for missing parameters
    required_params = required_params_map.get(gold_tool, [])
    for rp in required_params:
        if rp not in pred_params:
            return "missing_parameter"
            
    # Check for hallucinated or incorrect parameter values
    # If present but incorrect, let's see if it's hallucinated (not grounded in input)
    has_value_mismatch = False
    for p, val in pred_params.items():
        gold_val = gold_params.get(p)
        if gold_val is not None:
            if str(val).strip().lower() != str(gold_val).strip().lower():
                has_value_mismatch = True
                
                # Check grounding: is the generated value found in the original input text?
                # For non-string or short values (like integers), grounding is tricky, but let's check
                val_str = str(val).strip()
                if val_str and val_str.lower() not in input_text.lower():
                    return "hallucinated_parameter"
                    
    # If there's a value mismatch but it was somehow in the input text, 
    # it's still a parameter value error (we classify it as hallucinated_parameter in Phase 2 
    # as there is no wrong_field_value for tool calls)
    if has_value_mismatch:
        return "hallucinated_parameter"
        
    return ""  # No failure


def generate_failure_report(failures: List[str], is_phase2: bool = False) -> str:
    """Generates the markdown table representing failure classification counts and percentages.
    failures: list of failure category strings
    """
    categories = [
        "malformed_json",
        "schema_violation",
        "wrong_tool",
        "missing_parameter",
        "hallucinated_parameter",
        "wrong_field_value"
    ]
    
    # Filter out categories that don't apply to the phase
    if not is_phase2:
        categories.remove("wrong_tool")
        categories.remove("missing_parameter")
        categories.remove("hallucinated_parameter")
    else:
        categories.remove("wrong_field_value")
        
    counts = {cat: 0 for cat in categories}
    for f in failures:
        if f in counts:
            counts[f] += 1
            
    total_failures = len(failures)
    
    report_lines = [
        "| Failure Type | Count | % of Failures |",
        "|--------------------------|-------|----------------|"
    ]
    
    for cat in categories:
        count = counts[cat]
        pct = (count / total_failures * 100.0) if total_failures > 0 else 0.0
        report_lines.append(f"| {cat:<24} | {count:<5} | {pct:>12.1f}% |")
        
    return "\n".join(report_lines)
