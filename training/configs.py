import os
from typing import Dict, Any

# Base model to use
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VALIDATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Experiment configurations
EXPERIMENTS = {
    "zero_shot": {
        "description": "Base model zero-shot evaluation (no fine-tuning)",
        "is_finetuned": False,
        "lora_rank": 0,
        "lora_alpha": 0,
    },
    "qlora_r8": {
        "description": "QLoRA fine-tuned with LoRA rank 8",
        "is_finetuned": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "logging_steps": 10,
        "save_steps": 100,
    },
    "qlora_r16": {
        "description": "QLoRA fine-tuned with LoRA rank 16",
        "is_finetuned": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "logging_steps": 10,
        "save_steps": 100,
    }
}

def get_experiment_config(exp_name: str) -> Dict[str, Any]:
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment configuration: {exp_name}")
    return EXPERIMENTS[exp_name]
