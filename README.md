# 90-Day GenAI/LLM Challenge

This repo hosts daily *executable notes* (Markdown + notebooks + tiny demos) and small build tasks.
- **Theory days** live as research logs with a minimal, testable demo and a short notebook.
- **Build days** focus on shipping a small feature or benchmark with a clear README.

## Structure
```
genai-90day-challenge/
├── days/
│   ├── _TEMPLATE_THEORY/
│   ├── _TEMPLATE_BUILD/
│   └── day-12-attention-mechanism/        # example theory day
├── scripts/
│   └── new_day.py                         # generate a new day from templates
├── docs/
│   └── HOW_TO_WRITE_THEORY_DAY.md
├── .gitignore
├── LICENSE
└── CITATIONS.md
```

## Quickstart
```bash
# create a new theory day
python scripts/new_day.py --type theory --day 13 --slug "transformer-architecture"

# create a new build day
python scripts/new_day.py --type build --day 14 --slug "rag-bm25-vector"
```
