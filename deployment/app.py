"""
ReliaLM FastAPI Deployment Server

Endpoints:
  POST /predict   - Given input text, return structured output or tool call
  POST /evaluate  - Run evaluation suite against a provided test set
  GET  /metrics   - Return latest computed metrics
  GET  /health    - Basic health check
"""
import os
import sys
import json
import time
import random
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Path fix so the app can be launched from the deployment/ directory ──────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.evaluator import evaluate_phase1, evaluate_phase2
from evaluation.mock_executors import get_mock_executor
from failure_analysis.classifier import (
    classify_failure_phase1,
    classify_failure_phase2,
    generate_failure_report,
)

app = FastAPI(
    title="ReliaLM",
    description="Reliability Engineering Framework for Structured Output and Function-Calling",
    version="1.0.0",
)

# ── In-memory metrics store ─────────────────────────────────────────────────
_latest_metrics: Dict[str, Any] = {}

# ── Inference backend ────────────────────────────────────────────────────────
# Tries to load a real model if RELIALM_MODEL_DIR is set, otherwise falls back
# to the deterministic mock labeler so the API is always runnable.

def _build_inference_fn():
    model_dir = os.environ.get("RELIALM_MODEL_DIR", "")
    if not model_dir:
        # Fallback: use mock labeler
        from data.labeler import MockLabeler
        _mock = MockLabeler()
        def _infer(text: str, phase: int) -> str:
            if phase == 1:
                return json.dumps(_mock.label_issue_phase1(text))
            else:
                return json.dumps(_mock.label_issue_phase2(text))
        print("[ReliaLM] RELIALM_MODEL_DIR not set — using mock inference backend.")
        return _infer

    # Real model path
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_id = os.environ.get("RELIALM_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[ReliaLM] Loading base model: {base_model_id} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if device == "cuda":
            model_kwargs["load_in_4bit"] = True

        base = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
        print("[ReliaLM] Fine-tuned model loaded successfully.")

        from training.train import get_system_prompt

        def _infer_real(text: str, phase: int) -> str:
            sys_p = get_system_prompt(phase)
            prompt = (
                f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return gen.strip()

        return _infer_real

    except Exception as exc:
        print(f"[ReliaLM] Failed to load real model ({exc}). Falling back to mock.")
        from data.labeler import MockLabeler
        _mock2 = MockLabeler()
        def _infer_fallback(text: str, phase: int) -> str:
            if phase == 1:
                return json.dumps(_mock2.label_issue_phase1(text))
            return json.dumps(_mock2.label_issue_phase2(text))
        return _infer_fallback


_infer = _build_inference_fn()
_mock_exec = get_mock_executor()

# ── Request / Response models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    phase: int = 1  # 1 = structured output, 2 = function-calling

class PredictResponse(BaseModel):
    raw_output: str
    parsed: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EvaluateRequest(BaseModel):
    phase: int = 1
    examples: List[Dict[str, Any]]  # list of {"text": str, "label": dict}
    simulate: bool = False

class EvaluateResponse(BaseModel):
    metrics: Dict[str, Any]
    failure_report: str
    num_failures: int

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/metrics", tags=["Ops"])
def get_metrics():
    """Return the latest metrics computed by /evaluate."""
    if not _latest_metrics:
        # Try loading from the most recent experiment directory
        for exp in ["phase1_qlora_r16", "phase1_qlora_r8", "phase1_zero_shot"]:
            path = os.path.join(_ROOT, "experiments", exp, "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    return {"source": exp, "metrics": json.load(f)}
        raise HTTPException(status_code=404, detail="No metrics available yet. Run /evaluate first.")
    return {"source": "latest_evaluate_call", "metrics": _latest_metrics}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Given input text, return structured output (Phase 1) or tool call (Phase 2)."""
    if req.phase not in (1, 2):
        raise HTTPException(status_code=400, detail="phase must be 1 or 2")

    raw = _infer(req.text, req.phase)

    parsed = None
    error = None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        error = f"JSON parse error: {exc}"

    return PredictResponse(raw_output=raw, parsed=parsed, error=error)


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate(req: EvaluateRequest):
    """Run evaluation suite against a provided list of labelled examples."""
    global _latest_metrics

    if req.phase not in (1, 2):
        raise HTTPException(status_code=400, detail="phase must be 1 or 2")
    if not req.examples:
        raise HTTPException(status_code=400, detail="examples list is empty")

    texts = [ex["text"] for ex in req.examples]
    gold_labels = [ex["label"] for ex in req.examples]

    # Generate predictions
    predictions = []
    for text in texts:
        if req.simulate:
            # Return a perfect match for demo purposes when simulate=True
            predictions.append(json.dumps(gold_labels[texts.index(text)]))
        else:
            predictions.append(_infer(text, req.phase))

    # Compute metrics
    if req.phase == 1:
        metrics = evaluate_phase1(predictions, gold_labels)
        failures = [
            classify_failure_phase1(pred, gold, text)
            for pred, gold, text in zip(predictions, gold_labels, texts)
            if classify_failure_phase1(pred, gold, text)
        ]
        report = generate_failure_report(failures, is_phase2=False)
    else:
        metrics = evaluate_phase2(predictions, gold_labels, mock_executor=_mock_exec)
        failures = [
            classify_failure_phase2(pred, gold, text)
            for pred, gold, text in zip(predictions, gold_labels, texts)
            if classify_failure_phase2(pred, gold, text)
        ]
        report = generate_failure_report(failures, is_phase2=True)

    _latest_metrics = metrics
    return EvaluateResponse(metrics=metrics, failure_report=report, num_failures=len(failures))
