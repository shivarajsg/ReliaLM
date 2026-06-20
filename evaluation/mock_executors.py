from typing import Dict, Any, Tuple

def execute_search_repo(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Mock executor for search_repo.
    Parameters:
      - query: str (required)
      - language: str (optional, enum: python, javascript, go, java, any)
      - max_results: int (optional, default 10)
    """
    if "query" not in params:
        return False, "Missing required parameter: query"
        
    query = params["query"]
    if not isinstance(query, str) or not query.strip():
        return False, "Parameter 'query' must be a non-empty string"
        
    if "language" in params:
        lang = params["language"]
        valid_langs = {"python", "javascript", "go", "java", "any"}
        if not isinstance(lang, str) or lang.lower() not in valid_langs:
            return False, f"Parameter 'language' must be one of {valid_langs}, got '{lang}'"
            
    if "max_results" in params:
        max_res = params["max_results"]
        # Allow ints or strings that represent ints
        if isinstance(max_res, str):
            if max_res.isdigit():
                max_res = int(max_res)
            else:
                return False, "Parameter 'max_results' must be an integer"
        if not isinstance(max_res, int) or isinstance(max_res, bool):
            return False, "Parameter 'max_results' must be an integer"
            
    return True, "Mock search_repo: Search completed successfully"


def execute_read_docs(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Mock executor for read_docs.
    Parameters:
      - doc_path: str (required)
      - section: str (optional)
    """
    if "doc_path" not in params:
        return False, "Missing required parameter: doc_path"
        
    doc_path = params["doc_path"]
    if not isinstance(doc_path, str) or not doc_path.strip():
        return False, "Parameter 'doc_path' must be a non-empty string"
        
    if "section" in params:
        section = params["section"]
        if not isinstance(section, str):
            return False, "Parameter 'section' must be a string"
            
    return True, "Mock read_docs: Documentation section read successfully"


def execute_create_issue(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Mock executor for create_issue.
    Parameters:
      - title: str (required)
      - body: str (required)
      - labels: list of str (optional)
    """
    if "title" not in params:
        return False, "Missing required parameter: title"
    if "body" not in params:
        return False, "Missing required parameter: body"
        
    title = params["title"]
    body = params["body"]
    if not isinstance(title, str) or not title.strip():
        return False, "Parameter 'title' must be a non-empty string"
    if not isinstance(body, str) or not body.strip():
        return False, "Parameter 'body' must be a non-empty string"
        
    if "labels" in params:
        labels = params["labels"]
        if not isinstance(labels, list):
            return False, "Parameter 'labels' must be an array/list"
        for label in labels:
            if not isinstance(label, str):
                return False, f"All elements in 'labels' list must be strings, got: {type(label)}"
                
    return True, "Mock create_issue: Issue created successfully"


def execute_run_tests(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Mock executor for run_tests.
    Parameters:
      - test_path: str (required)
      - test_name: str (optional)
    """
    if "test_path" not in params:
        return False, "Missing required parameter: test_path"
        
    test_path = params["test_path"]
    if not isinstance(test_path, str) or not test_path.strip():
        return False, "Parameter 'test_path' must be a non-empty string"
        
    if "test_name" in params:
        test_name = params["test_name"]
        if not isinstance(test_name, str):
            return False, "Parameter 'test_name' must be a string"
            
    return True, "Mock run_tests: Tests executed successfully"


def execute_summarize_pr(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Mock executor for summarize_pr.
    Parameters:
      - pr_number: int (required)
    """
    if "pr_number" not in params:
        return False, "Missing required parameter: pr_number"
        
    pr_num = params["pr_number"]
    # Allow strings that represent ints
    if isinstance(pr_num, str):
        if pr_num.isdigit():
            pr_num = int(pr_num)
        else:
            return False, "Parameter 'pr_number' must be an integer"
            
    if not isinstance(pr_num, int) or isinstance(pr_num, bool):
        return False, "Parameter 'pr_number' must be an integer"
        
    return True, f"Mock summarize_pr: Pull request #{pr_num} summarized successfully"


def get_mock_executor():
    """Returns a function that executes any tool call dictionary.
    Ex: call = {"tool": "run_tests", "parameters": {"test_path": "test.py"}}
    """
    executors = {
        "search_repo": execute_search_repo,
        "read_docs": execute_read_docs,
        "create_issue": execute_create_issue,
        "run_tests": execute_run_tests,
        "summarize_pr": execute_summarize_pr
    }
    
    def executor(tool_call: Dict[str, Any]) -> Tuple[bool, str]:
        tool = tool_call.get("tool", "")
        params = tool_call.get("parameters", {})
        
        if tool not in executors:
            return False, f"Unknown tool: '{tool}'"
            
        return executors[tool](params)
        
    return executor
