import pytest
from data.schema import Phase1Output, ToolCall
from data.labeler import MockLabeler
from evaluation.evaluator import (
    extract_json_string,
    parse_and_validate_phase1,
    parse_and_validate_phase2,
    evaluate_phase1,
    evaluate_phase2
)
from failure_analysis.classifier import (
    classify_failure_phase1,
    classify_failure_phase2,
    generate_failure_report
)

def test_json_extraction():
    # Standard json block
    text1 = "Here is the result:\n```json\n{\"issue_type\": \"authentication\"}\n```\nHope it helps!"
    assert extract_json_string(text1) == "{\"issue_type\": \"authentication\"}"
    
    # Generic block
    text2 = "```\n{\"foo\": \"bar\"}\n```"
    assert extract_json_string(text2) == "{\"foo\": \"bar\"}"
    
    # Missing backticks but contains JSON
    text3 = "Random text {\"foo\": \"bar\"} random text"
    assert extract_json_string(text3) == "{\"foo\": \"bar\"}"


def test_phase1_schema_validation():
    # Valid data
    valid_data = {
        "issue_type": "authentication",
        "root_cause": "expired_jwt",
        "priority": "high",
        "affected_component": "login_service"
    }
    obj = Phase1Output(**valid_data)
    assert obj.priority == "high"
    
    # Invalid priority case
    invalid_data = valid_data.copy()
    invalid_data["priority"] = "extreme"
    with pytest.raises(ValueError):
        Phase1Output(**invalid_data)


def test_mock_labeler():
    labeler = MockLabeler()
    
    # Phase 1 labeling test
    p1 = labeler.label_issue_phase1("User JWT expired on login page")
    assert p1["issue_type"] == "authentication"
    assert p1["root_cause"] == "expired_jwt"
    
    # Phase 2 labeling test
    p2 = labeler.label_issue_phase2("Search authentication bugs in repo")
    assert p2["tool"] == "search_repo"
    assert "query" in p2["parameters"]


def test_evaluate_phase1():
    gold = [
        {"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "high", "affected_component": "login_service"},
        {"issue_type": "performance", "root_cause": "race_condition", "priority": "medium", "affected_component": "payment_gateway"}
    ]
    # Exact matches
    preds = [
        '{"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "high", "affected_component": "login_service"}',
        '{"issue_type": "performance", "root_cause": "race_condition", "priority": "medium", "affected_component": "payment_gateway"}'
    ]
    metrics = evaluate_phase1(preds, gold)
    assert metrics["json_validity"] == 100.0
    assert metrics["schema_compliance"] == 100.0
    assert metrics["exact_match"] == 100.0
    
    # Mismatch priority on first
    preds_bad = [
        '{"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "low", "affected_component": "login_service"}',
        '{"issue_type": "performance", "root_cause": "race_condition", "priority": "medium", "affected_component": "payment_gateway"}'
    ]
    metrics_bad = evaluate_phase1(preds_bad, gold)
    assert metrics_bad["json_validity"] == 100.0
    assert metrics_bad["schema_compliance"] == 100.0
    assert metrics_bad["exact_match"] == 50.0
    assert metrics_bad["field_accuracy"]["priority"] == 50.0


def test_evaluate_phase2():
    gold = [
        {"tool": "search_repo", "parameters": {"query": "auth bug"}},
        {"tool": "run_tests", "parameters": {"test_path": "tests/test_core.py"}}
    ]
    preds = [
        '{"tool": "search_repo", "parameters": {"query": "auth bug"}}',
        '{"tool": "run_tests", "parameters": {"test_path": "tests/test_core.py"}}'
    ]
    
    # We define a dummy executor that returns success for valid calls
    def dummy_executor(call):
        return True, "Success"
        
    metrics = evaluate_phase2(preds, gold, mock_executor=dummy_executor)
    assert metrics["json_validity"] == 100.0
    assert metrics["schema_compliance"] == 100.0
    assert metrics["tool_selection_accuracy"] == 100.0
    assert metrics["parameter_accuracy_aggregated"] == 100.0
    assert metrics["end_to_end_executability"] == 100.0


def test_failure_classifier_phase1():
    gold = {"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "high", "affected_component": "login_service"}
    input_text = "JWT expired token login"
    
    # Malformed JSON
    assert classify_failure_phase1("Not JSON at all", gold, input_text) == "malformed_json"
    
    # Schema violation (missing key)
    assert classify_failure_phase1('{"issue_type": "authentication"}', gold, input_text) == "schema_violation"
    
    # Wrong field value
    assert classify_failure_phase1(
        '{"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "low", "affected_component": "login_service"}',
        gold, input_text
    ) == "wrong_field_value"
    
    # Correct
    assert classify_failure_phase1(
        '{"issue_type": "authentication", "root_cause": "expired_jwt", "priority": "high", "affected_component": "login_service"}',
        gold, input_text
    ) == ""


def test_failure_classifier_phase2():
    gold = {"tool": "search_repo", "parameters": {"query": "auth bug"}}
    input_text = "Search auth bug in codebase"
    
    # Malformed JSON
    assert classify_failure_phase2("Not JSON", gold, input_text) == "malformed_json"
    
    # Schema violation
    assert classify_failure_phase2('{"tool": "search_repo"}', gold, input_text) == "schema_violation"
    
    # Wrong tool
    assert classify_failure_phase2('{"tool": "read_docs", "parameters": {"doc_path": "docs.md"}}', gold, input_text) == "wrong_tool"
    
    # Missing parameter
    assert classify_failure_phase2('{"tool": "search_repo", "parameters": {}}', gold, input_text) == "missing_parameter"
    
    # Hallucinated parameter (value mismatch + not in input)
    assert classify_failure_phase2('{"tool": "search_repo", "parameters": {"query": "invented_query"}}', gold, input_text) == "hallucinated_parameter"
    
    # Correct
    assert classify_failure_phase2('{"tool": "search_repo", "parameters": {"query": "auth bug"}}', gold, input_text) == ""


def test_failure_report_generation():
    failures = ["malformed_json", "schema_violation", "malformed_json"]
    report = generate_failure_report(failures, is_phase2=False)
    assert "malformed_json" in report
    assert "schema_violation" in report
    assert "2" in report  # Count for malformed_json


def test_mock_executors():
    from evaluation.mock_executors import get_mock_executor
    exec_func = get_mock_executor()
    
    # Test search_repo
    res, msg = exec_func({"tool": "search_repo", "parameters": {"query": "authentication"}})
    assert res is True
    
    res, msg = exec_func({"tool": "search_repo", "parameters": {}})
    assert res is False
    assert "Missing required parameter" in msg
    
    # Test read_docs
    res, msg = exec_func({"tool": "read_docs", "parameters": {"doc_path": "README.md"}})
    assert res is True
    
    # Test create_issue
    res, msg = exec_func({"tool": "create_issue", "parameters": {"title": "title", "body": "body"}})
    assert res is True
    
    # Test run_tests
    res, msg = exec_func({"tool": "run_tests", "parameters": {"test_path": "tests/test_core.py"}})
    assert res is True
    
    # Test summarize_pr
    res, msg = exec_func({"tool": "summarize_pr", "parameters": {"pr_number": 123}})
    assert res is True
    
    res, msg = exec_func({"tool": "summarize_pr", "parameters": {"pr_number": "not_an_int"}})
    assert res is False
