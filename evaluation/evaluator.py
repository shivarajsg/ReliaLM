import json
import re
from typing import Dict, Any, List, Tuple, Optional
from data.schema import Phase1Output, ToolCall

# Regex to extract JSON block from markdown
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def extract_json_string(raw_text: str) -> str:
    """Extracts JSON string from raw text, stripping markdown code blocks if present."""
    if not raw_text:
        return ""
    
    text = raw_text.strip()
    
    # Try finding markdown JSON block
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
        
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
        
    return text


def parse_and_validate_phase1(raw_output: str) -> Tuple[bool, bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parses raw text as Phase 1 JSON.
    Returns:
        (is_valid_json, is_schema_compliant, parsed_dict, error_message)
    """
    json_str = extract_json_string(raw_output)
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        return False, False, None, f"JSON parse error: {str(e)}"
        
    # Check schema compliance
    if not isinstance(parsed, dict):
        return True, False, parsed, "Parsed JSON is not a dictionary"
        
    required_keys = {"issue_type", "root_cause", "priority", "affected_component"}
    missing_keys = required_keys - parsed.keys()
    if missing_keys:
        return True, False, parsed, f"Missing required keys: {missing_keys}"
        
    # Validate types and values using Pydantic
    try:
        Phase1Output(**parsed)
    except Exception as e:
        return True, False, parsed, f"Schema validation error: {str(e)}"
        
    return True, True, parsed, None


def parse_and_validate_phase2(raw_output: str) -> Tuple[bool, bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parses raw text as Phase 2 tool call.
    Returns:
        (is_valid_json, is_schema_compliant, parsed_dict, error_message)
    """
    json_str = extract_json_string(raw_output)
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        return False, False, None, f"JSON parse error: {str(e)}"
        
    if not isinstance(parsed, dict):
        return True, False, parsed, "Parsed JSON is not a dictionary"
        
    if "tool" not in parsed or "parameters" not in parsed:
        return True, False, parsed, "Missing 'tool' or 'parameters' keys"
        
    if not isinstance(parsed["parameters"], dict):
        return True, False, parsed, "'parameters' must be a dictionary"
        
    return True, True, parsed, None


# ==========================================
# Phase 1 Evaluation Metrics
# ==========================================

def evaluate_phase1(predictions: List[str], gold_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes Phase 1 metrics.
    predictions: Raw output strings from model
    gold_labels: List of gold dicts
    """
    total = len(gold_labels)
    if total == 0:
        return {}
        
    valid_json_count = 0
    schema_compliant_count = 0
    exact_match_count = 0
    
    # Per-field exact match counters
    fields = ["issue_type", "root_cause", "priority", "affected_component"]
    field_correct_counts = {f: 0 for f in fields}
    
    for pred, gold in zip(predictions, gold_labels):
        is_json, is_schema, parsed, _ = parse_and_validate_phase1(pred)
        
        if is_json:
            valid_json_count += 1
            
        if is_schema and parsed:
            schema_compliant_count += 1
            
            # Compare fields (case-insensitive where applicable, let's stick to exact match as per spec)
            all_match = True
            for f in fields:
                gold_val = str(gold.get(f, "")).strip().lower()
                pred_val = str(parsed.get(f, "")).strip().lower()
                if gold_val == pred_val:
                    field_correct_counts[f] += 1
                else:
                    all_match = False
            
            if all_match:
                exact_match_count += 1
                
    return {
        "json_validity": (valid_json_count / total) * 100.0,
        "schema_compliance": (schema_compliant_count / total) * 100.0,
        "field_accuracy": {f: (field_correct_counts[f] / total) * 100.0 for f in fields},
        "exact_match": (exact_match_count / total) * 100.0,
        "total_examples": total
    }


# ==========================================
# Phase 2 Evaluation Metrics
# ==========================================

def evaluate_phase2(predictions: List[str], gold_labels: List[Dict[str, Any]], mock_executor=None) -> Dict[str, Any]:
    """Computes Phase 2 metrics.
    predictions: Raw output strings from model
    gold_labels: List of gold tool call dicts (keys: 'tool', 'parameters')
    mock_executor: Optional function/object that executes tool calls
    """
    total = len(gold_labels)
    if total == 0:
        return {}
        
    valid_json_count = 0
    schema_compliant_count = 0
    tool_select_correct = 0
    
    # Required parameters per tool
    required_params = {
        "search_repo": ["query"],
        "read_docs": ["doc_path"],
        "create_issue": ["title", "body"],
        "run_tests": ["test_path"],
        "summarize_pr": ["pr_number"]
    }
    
    # We count total required parameter matches across all examples
    total_req_params = 0
    correct_req_params = 0
    
    # Per-parameter metric counts
    param_counts = {}  # param_name -> {"total": int, "correct": int}
    
    executable_count = 0
    
    for pred, gold in zip(predictions, gold_labels):
        gold_tool = gold.get("tool", "")
        gold_params = gold.get("parameters", {})
        
        # Count required parameters for this gold tool
        reqs = required_params.get(gold_tool, [])
        for p in reqs:
            total_req_params += 1
            param_counts.setdefault(p, {"total": 0, "correct": 0})
            param_counts[p]["total"] += 1
            
        is_json, is_schema, parsed, _ = parse_and_validate_phase2(pred)
        
        if is_json:
            valid_json_count += 1
            
        is_correct_tool = False
        if is_schema and parsed:
            schema_compliant_count += 1
            pred_tool = parsed.get("tool", "")
            pred_params = parsed.get("parameters", {})
            
            # 1. Tool Selection Accuracy
            if pred_tool == gold_tool:
                tool_select_correct += 1
                is_correct_tool = True
                
            # 2. Parameter Accuracy (only check required parameters of gold tool)
            for p in reqs:
                gold_p_val = gold_params.get(p)
                pred_p_val = pred_params.get(p) if is_correct_tool else None
                
                # Check if correct (exact string/int/type comparison)
                # Convert to string to compare robustly (e.g. pr_number can be str or int, handle appropriately)
                if pred_p_val is not None:
                    # Match types if possible, or convert to string for basic check
                    # Let's check exact match
                    if str(pred_p_val).strip().lower() == str(gold_p_val).strip().lower():
                        correct_req_params += 1
                        param_counts[p]["correct"] += 1
                        
            # 3. End-to-End Executability
            if mock_executor:
                try:
                    success, _ = mock_executor(parsed)
                    if success:
                        executable_count += 1
                except Exception:
                    pass
        else:
            # If JSON is not valid or not schema compliant, it is not executable
            pass
            
    # Compute per-parameter accuracy dict
    per_param_accuracy = {}
    for p, counts in param_counts.items():
        if counts["total"] > 0:
            per_param_accuracy[p] = (counts["correct"] / counts["total"]) * 100.0
            
    return {
        "json_validity": (valid_json_count / total) * 100.0,
        "schema_compliance": (schema_compliant_count / total) * 100.0,
        "tool_selection_accuracy": (tool_select_correct / total) * 100.0,
        "parameter_accuracy_aggregated": (correct_req_params / total_req_params * 100.0) if total_req_params > 0 else 0.0,
        "parameter_accuracy_per_param": per_param_accuracy,
        "end_to_end_executability": (executable_count / total) * 100.0,
        "total_examples": total
    }
