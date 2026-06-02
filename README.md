# Responsible Minerals Compliance Assistant

RAG agent for conflict minerals compliance (EU 2017/821, OECD Guidance, CSDDD).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/popovychiryna/Responsible-Minerals-Compliance-Assistant.git
cd Responsible-Minerals-Compliance-Assistant

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Ollama and pull models (first time only)
ollama serve &
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b

# 4. Add your documents to the data/ folder
#    Supported formats: .pdf, .html, .htm, .md, .txt
```

## Run

```bash
# Live demo (success + abstain + blocked)
python run_eval.py --demo

# Full evaluation → results/eval_results.json + graphs
python run_eval.py --eval

# Single question
python run_eval.py --query "What are the key obligations under EU 2017/821?"
```

## Project structure

```
├── run_eval.py          # main pipeline
├── requirements.txt
├── data/                # knowledge base documents (add your PDFs here)
├── results/             # eval_results.json + graphs (auto-created)
└── test_suite.json      # test questions with expected answers
```

## Environment variables (optional overrides)

| Variable       | Default                              |
|----------------|--------------------------------------|
| `OLLAMA_URL`   | `http://localhost:11434`             |
| `AGENT_MODEL`  | `qwen2.5:7b`                         |
| `JUDGE_MODEL`  | `qwen2.5:14b`                        |
| `EMBED_MODEL`  | `BAAI/bge-small-en-v1.5`             |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
