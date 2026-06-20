# ReliaLM: Reliability Engineering Framework for Structured Output and Function-Calling

ReliaLM is a reliability engineering framework designed to measure, evaluate, and improve the reliability of structured JSON outputs and function-calling generation in open-weight LLMs. 

## Research Question
How reliably can an open-weight LLM generate structured outputs and function calls, and how much does QLoRA fine-tuning improve that reliability?

## Project Architecture
The project is structured as follows:
- **`data/`**: Logic for loading public GitHub issues and prompting LLMs to label them.
- **`datasets/`**: Data splits (train, validation, and the frozen gold test sets).
- **`training/`**: QLoRA training scripts and parameter configurations.
- **`evaluation/`**: Metrics computation and Python mock executors.
- **`failure_analysis/`**: A rule-based failure classifier and analysis tables.
- **`deployment/`**: FastAPI backend serving predictions and evaluations.
- **`experiments/`**: Log files, model configurations, and experiment results.
- **`report/`**: The final written engineering report.

## Setup Instructions
1. Clone the repository and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables in a `.env` file (e.g., API keys for data labeling).
3. Follow the execution steps detailed in the report.
