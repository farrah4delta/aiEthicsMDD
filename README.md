# AIEI L4 Evaluator

Automatically evaluate the performance of two chat-based LLMs on four AIEI L4 indicators and generate structured comparison reports (JSON + Markdown).

## Evaluation Indicators

This project evaluates the following four L4 indicators (Dev-oriented):

1. **L4_DEV_SafetyObjectives** – Safety objectives & value trade-offs are articulated
2. **L4_DEV_RedTeamingResults** – Red-teaming results inform alignment updates
3. **L4_DEV_DomainSpecificFine** – Domain-specific fine-tunes documented with safety constraints
4. **L4_DEV_ModelUpdateRollback** – Model update and rollback mechanisms exist

## Installation

```bash
# Clone the project
cd 680_L4s

# Install dependencies
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit the `.env` file and fill in your API keys and configuration:
```env
# API provider (default: openai)
API_PROVIDER=openai

# OpenAI API Key (required when using OpenAI)
OPENAI_API_KEY=sk-...

# Gemini API Key (required when using Gemini, optional)
# GEMINI_API_KEY=your-gemini-api-key-here

# Model configuration
MODEL_A_NAME=gpt-4o-mini
MODEL_B_NAME=gemini-2.5-flash

# Evaluation configuration
TEMPERATURE=0.2
MAX_RETRIES=3
USE_LLM_JUDGE=true

# LLM Judge configuration (optional)
JUDGE_MODEL=gpt-4o-2024-08-06
```

**Note**:
- Default uses OpenAI API. To use Gemini, set `API_PROVIDER=gemini` and configure `GEMINI_API_KEY`
- You can mix different providers: specify different API providers for the two models via `MODEL_A_PROVIDER` and `MODEL_B_PROVIDER`
- All configuration items can be set via environment variables or `.env` file
- For detailed configuration instructions, refer to the `.env.example` file

## Usage

### Command Line

```bash
python scripts/run_evaluation.py \
    --model-a gpt-4 \
    --model-b gpt-3.5-turbo \
    --output-json Reports/report.json \
    --output-md Reports/report.md
```

### As a Python Module

```python
from aiei_l4.config import get_settings
from aiei_l4.pipeline import run_full_evaluation
from aiei_l4.report import to_json, to_markdown

settings = get_settings()
result = run_full_evaluation(settings)

# Generate reports
json_report = to_json(result)
md_report = to_markdown(result)

print(md_report)
```

## Project Structure

```
aiei_l4_evaluator/
├── aiei_l4/           # Core modules
│   ├── __init__.py
│   ├── config.py      # Configuration management
│   ├── models.py      # Model clients
│   ├── prompts.py     # Prompt templates
│   ├── scoring.py     # Scoring logic
│   ├── pipeline.py    # Evaluation pipeline
│   ├── report.py      # Report generation
│   └── utils.py       # Utility functions
├── scripts/           # Scripts
│   └── run_evaluation.py
├── tests/             # Tests
│   ├── test_prompts.py
│   ├── test_scoring.py
│   └── test_pipeline.py
├── pyproject.toml
├── README.md
└── .env.example
```

## Scoring Mechanism

Each indicator score range: 0-4 points

Scoring methods:
1. **Rule-based scoring**: Based on keywords, number of examples, structure level, etc.
2. **LLM Judge (optional)**: Use a judge model to score responses

## Output

After evaluation completes, the following files will be generated in the `Reports/` folder:
- **JSON report**: Contains complete evaluation results and raw responses (default: `Reports/report.json`)
- **Markdown report**: Human-readable comparison report with summary table and detailed analysis (default: `Reports/report.md`)

## Development

Run tests:
```bash
pytest
```

Code formatting:
```bash
black aiei_l4 scripts tests
ruff check aiei_l4 scripts tests
```

## License

MIT
