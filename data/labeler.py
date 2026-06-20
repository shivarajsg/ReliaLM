import os
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLabeler(ABC):
    """Abstract base class for data labelers."""
    
    @abstractmethod
    def label_issue_phase1(self, issue_text: str) -> Dict[str, Any]:
        """Generate Phase 1 structured output (JSON matching schema)."""
        pass
        
    @abstractmethod
    def label_issue_phase2(self, issue_text: str) -> Dict[str, Any]:
        """Generate Phase 2 function call (JSON matching tool schema)."""
        pass


class MockLabeler(BaseLabeler):
    """Deterministic mock labeler for testing and pipeline validation without API keys."""
    
    def __init__(self):
        self.issue_types = ["authentication", "performance", "ui_bug", "data_loss", "database"]
        self.root_causes = ["expired_jwt", "race_condition", "null_pointer", "query_timeout", "css_misalignment"]
        self.priorities = ["low", "medium", "high", "critical"]
        self.components = ["login_service", "payment_gateway", "user_db", "frontend_header", "auth_middleware"]
        
        self.tools = ["search_repo", "read_docs", "create_issue", "run_tests", "summarize_pr"]

    def label_issue_phase1(self, issue_text: str) -> Dict[str, Any]:
        # Simple heuristic to make it feel somewhat realistic
        text = issue_text.lower()
        
        issue_type = "ui_bug"
        if any(w in text for w in ["login", "jwt", "auth", "permission"]):
            issue_type = "authentication"
        elif any(w in text for w in ["slow", "timeout", "latency", "hang"]):
            issue_type = "performance"
        elif any(w in text for w in ["delete", "drop", "lost", "leak"]):
            issue_type = "data_loss"
        elif any(w in text for w in ["db", "query", "sql"]):
            issue_type = "database"
            
        root_cause = "null_pointer"
        if "jwt" in text:
            root_cause = "expired_jwt"
        elif "race" in text or "concurrency" in text:
            root_cause = "race_condition"
        elif "timeout" in text or "slow" in text:
            root_cause = "query_timeout"
        elif "align" in text or "css" in text or "color" in text:
            root_cause = "css_misalignment"
            
        priority = "medium"
        if any(w in text for w in ["urgent", "critical", "broken", "down"]):
            priority = "critical"
        elif any(w in text for w in ["high", "blocker"]):
            priority = "high"
        elif any(w in text for w in ["minor", "low", "docs"]):
            priority = "low"
            
        affected_component = "frontend_header"
        if "login" in text or "auth" in text:
            affected_component = "login_service"
        elif "pay" in text or "stripe" in text:
            affected_component = "payment_gateway"
        elif "db" in text or "user" in text:
            affected_component = "user_db"
        elif "middleware" in text:
            affected_component = "auth_middleware"
            
        return {
            "issue_type": issue_type,
            "root_cause": root_cause,
            "priority": priority,
            "affected_component": affected_component
        }

    def label_issue_phase2(self, issue_text: str) -> Dict[str, Any]:
        text = issue_text.lower()
        
        if "search" in text or "find" in text or "grep" in text:
            q = text.replace("search", "").replace("find", "").replace("grep", "").strip()
            return {
                "tool": "search_repo",
                "parameters": {
                    "query": q if q else "test",
                    "language": "python" if "python" in text else ("javascript" if "javascript" in text else "any"),
                    "max_results": 10
                }
            }
        elif "read" in text or "doc" in text or "guide" in text:
            return {
                "tool": "read_docs",
                "parameters": {
                    "doc_path": "docs/setup.md" if "setup" in text else "README.md",
                    "section": "installation" if "install" in text else "introduction"
                }
            }
        elif "create" in text or "issue" in text or "report" in text:
            return {
                "tool": "create_issue",
                "parameters": {
                    "title": f"Bug: {issue_text[:30]}...",
                    "body": issue_text,
                    "labels": ["bug"]
                }
            }
        elif "test" in text or "run" in text:
            return {
                "tool": "run_tests",
                "parameters": {
                    "test_path": "tests/test_auth.py" if "auth" in text else "tests/test_core.py",
                    "test_name": "test_login" if "login" in text else "test_all"
                }
            }
        elif "pr" in text or "pull" in text or "merge" in text:
            # Extract number
            pr_num = 42
            for word in text.split():
                if word.isdigit():
                    pr_num = int(word)
                    break
            return {
                "tool": "summarize_pr",
                "parameters": {
                    "pr_number": pr_num
                }
            }
        else:
            # Default
            return {
                "tool": "search_repo",
                "parameters": {"query": issue_text[:50]}
            }


class OpenAILabeler(BaseLabeler):
    """Labeler using OpenAI API or OpenAI-compatible backends (Groq, local, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        if not self.api_key and not self.base_url:
            raise ValueError("API key must be provided or set via environment variable.")
            
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_llm(self, prompt: str, system_prompt: str, json_mode: bool = True) -> str:
        response_format = {"type": "json_object"} if json_mode else None
        
        # Add retry logic
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=response_format,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                print(f"LLM API call failed (attempt {attempt + 1}/3): {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to get response from LLM API after 3 attempts.")

    def label_issue_phase1(self, issue_text: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a helpful labeling assistant for a software reliability framework.\n"
            "Given a natural-language description of a software issue, extract details matching the fixed JSON schema:\n"
            "{\n"
            "  \"issue_type\": string, // e.g. \"authentication\", \"performance\", \"ui_bug\", \"data_loss\", \"database\"\n"
            "  \"root_cause\": string, // short phrase, e.g. \"expired_jwt\", \"race_condition\", \"null_pointer\", \"query_timeout\"\n"
            "  \"priority\": string, // exactly one of: \"low\", \"medium\", \"high\", \"critical\"\n"
            "  \"affected_component\": string // e.g. \"login_service\", \"payment_gateway\", \"user_db\"\n"
            "}\n"
            "Return ONLY a valid JSON object matching this schema. Do not output markdown code blocks. Do not invent details."
        )
        
        prompt = f"Software Issue Description: \"{issue_text}\"\n\nJSON Output:"
        response_str = self._call_llm(prompt, system_prompt, json_mode=True)
        return json.loads(response_str)

    def label_issue_phase2(self, issue_text: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an assistant that translates natural language requests into structured tool calls.\n"
            "You must select exactly one of the following 5 tools:\n"
            "1. search_repo(query: string, language: enum['python','javascript','go','java','any'], max_results: integer)\n"
            "2. read_docs(doc_path: string, section: string)\n"
            "3. create_issue(title: string, body: string, labels: array[string])\n"
            "4. run_tests(test_path: string, test_name: string)\n"
            "5. summarize_pr(pr_number: integer)\n\n"
            "Format the output as a single JSON object with the following structure:\n"
            "{\n"
            "  \"tool\": string, // name of the selected tool\n"
            "  \"parameters\": object // arguments matching the tool parameters\n"
            "}\n"
            "Return ONLY a valid JSON object. Do not include markdown code blocks. Make sure required parameters are present."
        )
        
        prompt = f"Request: \"{issue_text}\"\n\nJSON Output:"
        response_str = self._call_llm(prompt, system_prompt, json_mode=True)
        return json.loads(response_str)


class GeminiLabeler(BaseLabeler):
    """Labeler using Gemini API via OpenAI compatibility interface or Google Generative AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided.")
        
        # We use the official OpenAI compatibility base URL for Gemini to keep it simple and unified
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = "gemini-1.5-flash"
        
        self.openai_backend = OpenAILabeler(api_key=self.api_key, base_url=base_url, model=model)

    def label_issue_phase1(self, issue_text: str) -> Dict[str, Any]:
        return self.openai_backend.label_issue_phase1(issue_text)

    def label_issue_phase2(self, issue_text: str) -> Dict[str, Any]:
        return self.openai_backend.label_issue_phase2(issue_text)


def get_labeler(provider: str, **kwargs) -> BaseLabeler:
    provider = provider.lower()
    if provider == "mock":
        return MockLabeler()
    elif provider == "openai":
        return OpenAILabeler(**kwargs)
    elif provider == "gemini":
        return GeminiLabeler(**kwargs)
    else:
        raise ValueError(f"Unknown labeling provider: {provider}")
