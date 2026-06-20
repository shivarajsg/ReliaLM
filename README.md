# ReliaLM: Reliability Engineering Framework for Structured Output and Function-Calling in LLM Agents

ReliaLM is a reliability engineering framework designed to measure, evaluate, and improve the reliability of structured JSON outputs and function-calling generation in open-weight LLM agents (specifically `Qwen2.5-7B-Instruct`). 

This project studies the core problem of agent failures due to unusable outputs (e.g., malformed JSON, wrong tool invocations, and hallucinated parameters) and compares zero-shot baseline performance against QLoRA fine-tuned adapters (ranks 8 and 16).

---

## 🚀 Key Features

* **Configurable Labeler Interface**: Generate training/testing datasets using pluggable LLM provider backends (OpenAI-compatible APIs, Gemini, Groq, local Hugging Face, or Mock).
* **Double-Phase Pipeline**:
  - **Phase 1 (Structured Outputs)**: Extracts software issue metadata into a flat JSON schema.
  - **Phase 2 (Function-Calling)**: Translates natural language requests into valid JSON tool calls.
* **Deterministic Verification**: Integrated Python mock executors for 5 tools (`search_repo`, `read_docs`, `create_issue`, `run_tests`, `summarize_pr`) verifying execution correctness.
* **Failure Classifier Engine**: Automatically categories errors into structural (`malformed_json`, `schema_violation`) or semantic (`wrong_tool`, `hallucinated_parameter`, `wrong_field_value`) failure modes.
* **FastAPI Microservice**: High-performance production API serving predictions and evaluation triggers, containerized with Docker.

---

## 📁 Repository Structure

```
ReliaLM/
├── data/                  # Raw issue loading and LLM dataset labelling logic
├── datasets/              # Train, validation, and frozen gold test splits
├── deployment/            # FastAPI deployment server and Dockerfile
├── evaluation/            # Metrics computation and Python mock tool executors
├── experiments/           # Simulation configs, logs, metrics, and failure analysis markdown
├── failure_analysis/      # Failure classifier engine and report generators
├── notebooks/             # Scrap/run templates for Kaggle or RunPod
├── report/                # Final written comparative engineering report
├── tests/                 # Core test suite and API integration tests
├── requirements.txt       # Project dependencies
└── README.md              # Project README
```

---

## 🛠️ Installation & Setup

1. **Clone the repository and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   # API Keys for the data labeler pipeline (e.g., OpenAI, Gemini, Groq)
   OPENAI_API_KEY=your-openai-api-key
   GEMINI_API_KEY=your-gemini-api-key
   GROQ_API_KEY=your-groq-api-key

   # Model Selection configuration
   RELIALM_MODEL_DIR=experiments/phase2_qlora_r16
   RELIALM_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
   ```

3. **Verify Installation**:
   Run the pytest test suite to ensure all components, mock environments, and FastAPI server endpoints are fully operational:
   ```bash
   python -m pytest tests/ -v
   ```

---

## 📊 Running the Framework

### 1. Dataset Generation
Generate the Phase 1 or Phase 2 training, validation, and gold test sets. To run using a mock labeler (without using API keys):
```bash
# Generate Phase 1 datasets
python -m data.generate_datasets --phase 1 --provider mock

# Generate Phase 2 datasets
python -m data.generate_datasets --phase 2 --provider mock
```

### 2. Fine-Tuning (QLoRA)
Execute fine-tuning on a GPU-enabled instance. Run in `--validation-mode` on a CPU to verify the pipeline:
```bash
# Pipeline validation on CPU (few steps, tiny model)
python -m training.train --phase 1 --config qlora_r8 --validation-mode

# Full training run (requires GPU)
python -m training.train --phase 2 --config qlora_r16
```

### 3. Evaluation and Simulations
Compute metrics and failure analysis tables against the frozen gold test sets. To force simulation mode:
```bash
# Run evaluations for zero-shot baseline
python -m evaluation.run_eval --phase 1 --experiment zero_shot --simulate
python -m evaluation.run_eval --phase 2 --experiment zero_shot --simulate

# Run evaluations for QLoRA Rank 8 & 16
python -m evaluation.run_eval --phase 2 --experiment qlora_r8 --simulate
python -m evaluation.run_eval --phase 2 --experiment qlora_r16 --simulate
```

### 4. Running the FastAPI Deployment
Launch the API server using Uvicorn:
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```
Access the interactive OpenAPI Swagger documentation at `http://localhost:8000/docs`.

To build and run inside a Docker container:
```bash
docker build -t relialm:latest -f deployment/Dockerfile .
docker run -p 8000:8000 relialm:latest
```

---

## 📈 Summary of Experimental Results

Below is a summary of the comparative analysis between Zero-Shot baseline and QLoRA fine-tuning (Rank 8 & 16) evaluated on the 300-example frozen Gold Test sets.

### Phase 1: Structured JSON Output
* Fine-tuning eliminated ~94% of formatting and syntax errors, raising JSON validity from **78.33%** (Zero-Shot) to **98.67%** (Rank 16).
* Exact Match accuracy surged from **16.00%** to **77.67%**.

### Phase 2: Function-Calling / Tool Execution
* Zero-shot models are highly unreliable, completing only **52.33%** of executions.
* **QLoRA Rank 16** dramatically out-performed **Rank 8**, achieving **97.00% Tool Selection Accuracy** (vs. 83.67%) and **92.26% Aggregated Parameter Accuracy** (vs. 76.19%), showing that higher capacity adapters are required for logical tool parameter binding.

For a detailed breakdown of failure modes and quantitative tables, read the full [Engineering Report](file:///c:/Users/KumarShiva/Downloads/ReliaLM%20-%20Reliability%20Engineering%20Framework/report/report.md).
