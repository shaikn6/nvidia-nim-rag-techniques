# Contributing

Thanks for taking the time to contribute.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # NVIDIA_NIM_API_KEY is not needed for the tests or the retrieval eval

pytest --cov=src --cov-report=term-missing --cov-fail-under=95   # what CI runs
ruff check src/ tests/ --select E,W,F --ignore E501              # what CI runs
```

On macOS, `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` because `faiss-cpu` and `torch` both bundle
an OpenMP runtime; prefix other commands with it if they abort.

## Ground rules

- **One logical change per PR.** Keep diffs reviewable.
- **Tests are not optional.** New behaviour needs a test; bug fixes need a test that fails before the fix.
  CI fails below 95% line coverage on `src/`.
- **No network in tests.** Tests use the demo embeddings and stubs; nothing may call NVIDIA NIM.
- **Retrieval numbers come from the code.** If a change moves the retrieval-quality table, re-run
  `python -m src.pipeline.retrieval_eval` and update the README from `eval/retrieval_results.json` in the same PR.
- **No secrets in code.** Keys come from the environment (`.env`, never committed).

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Reporting bugs / proposing features

Use the issue templates. For anything security-related, follow [SECURITY.md](SECURITY.md) instead of opening
a public issue.
