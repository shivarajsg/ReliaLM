import os
import sys
import json
import torch
import argparse
from typing import Dict, Any
# Case-insensitive sys.path filtering to prevent local 'datasets' folder shadowing
def import_datasets():
    import sys
    import os
    original_path = list(sys.path)
    try:
        workspace_dir = os.path.abspath(os.getcwd()).lower()
        sys.path = [
            p for p in sys.path 
            if p not in ("", ".") 
            and os.path.abspath(p).lower() != workspace_dir
        ]
        from datasets import load_dataset
        return load_dataset
    finally:
        sys.path = original_path

load_dataset = import_datasets()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from training.configs import get_experiment_config, BASE_MODEL, VALIDATION_MODEL

# Prompt formats matching training templates
def get_system_prompt(phase: int) -> str:
    if phase == 1:
        return (
            "You are a helpful software engineering agent. Extract details from the software issue description "
            "and output a JSON object adhering to this schema:\n"
            "{\n"
            "  \"issue_type\": string,\n"
            "  \"root_cause\": string,\n"
            "  \"priority\": string,\n"
            "  \"affected_component\": string\n"
            "}"
        )
    else:
        return (
            "You are a helpful software engineering agent. Translate the natural language request into a "
            "JSON tool call conforming to the schema of one of the available tools:\n"
            "- search_repo(query, language, max_results)\n"
            "- read_docs(doc_path, section)\n"
            "- create_issue(title, body, labels)\n"
            "- run_tests(test_path, test_name)\n"
            "- summarize_pr(pr_number)\n"
            "Output JSON: {\"tool\": \"...\", \"parameters\": {...}}"
        )


def format_sample(item: Dict[str, Any], phase: int) -> Dict[str, Any]:
    system_prompt = get_system_prompt(phase)
    user_prompt = item["text"]
    response = json.dumps(item["label"])
    
    formatted = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return {"text_formatted": formatted}


def main():
    parser = argparse.ArgumentParser(description="ReliaLM SFT QLoRA Trainer")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2], help="Phase 1 (Structured Output) or Phase 2 (Function-Calling)")
    parser.add_argument("--config", type=str, required=True, choices=["qlora_r8", "qlora_r16"], help="Experiment config name")
    parser.add_argument("--validation-mode", action="store_true", help="Run in pipeline validation mode (small model, few steps, CPU friendly)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model checkpoints")
    
    args = parser.parse_args()
    
    # Load config
    config = get_experiment_config(args.config)
    print(f"Loaded config: {args.config} - {config['description']}")
    
    # Set up hardware options
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model Selection
    model_id = VALIDATION_MODEL if (args.validation_mode or device == "cpu") else BASE_MODEL
    print(f"Target model ID: {model_id}")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Dataset paths
    train_path = os.path.abspath(f"datasets/phase{args.phase}/train.jsonl")
    val_path = os.path.abspath(f"datasets/phase{args.phase}/val.jsonl")
    
    print(f"Loading datasets:\n  Train: {train_path}\n  Val: {val_path}")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Train or validation JSONL file not found. Run dataset generation first.")
        sys.exit(1)
        
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    
    # Limit dataset sizes in validation mode
    if args.validation_mode:
        print("Validation mode: slicing datasets to 10 train and 5 val examples.")
        dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(5, len(dataset["validation"]))))
        
    # Map prompt formatter
    dataset = dataset.map(lambda x: format_sample(x, args.phase))
    
    # Weights & Biases Configuration
    if os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_PROJECT"] = "ReliaLM"
        report_to = "wandb"
        print("Weights & Biases logging enabled.")
    else:
        report_to = "none"
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B API key not found. Logging locally only.")
        
    # Quantization settings (only for GPU training)
    if device == "cuda":
        print("Configuring 4-bit BitsAndBytes quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if args.validation_mode else torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        print("CUDA not available. Disabling 4-bit quantization.")
        bnb_config = None
        
    # Load Model
    print("Loading base model...")
    model_kwargs = {}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu" if device == "cpu" else "auto"
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Prepare model for PEFT
    if device == "cuda" and bnb_config:
        model = prepare_model_for_kbit_training(model)
        
    # LoRA config
    # Target modules commonly present in Qwen models
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Output directory
    output_dir = args.output_dir or f"experiments/phase{args.phase}_{args.config}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Override epochs/steps for CPU/validation run
    epochs = 1 if args.validation_mode else config.get("num_epochs", 3)
    max_steps = 2 if args.validation_mode else -1
    
    print("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=1 if args.validation_mode else config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=1 if args.validation_mode else 2,
        gradient_accumulation_steps=1 if args.validation_mode else config.get("gradient_accumulation_steps", 8),
        evaluation_strategy="epoch" if max_steps == -1 else "no",
        save_strategy="epoch" if max_steps == -1 else "no",
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=0.01,
        fp16=(device == "cuda" and args.validation_mode),
        bf16=(device == "cuda" and not args.validation_mode),
        logging_steps=1 if args.validation_mode else config.get("logging_steps", 10),
        save_total_limit=1,
        report_to=report_to,
        gradient_checkpointing=(device == "cuda"),
        use_cpu=(device == "cpu"),
        remove_unused_columns=True
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text_formatted",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )
    
    print("Starting training run...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter checkpoints to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training job complete!")


if __name__ == "__main__":
    main()
