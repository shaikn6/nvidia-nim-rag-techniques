## What

<!-- One or two sentences on the change. -->

## Why

<!-- The problem this solves or the behaviour it adds. -->

## Checklist

- [ ] `ruff check src/ tests/ --select E,W,F --ignore E501` passes
- [ ] `pytest --cov=src --cov-fail-under=95` passes
- [ ] New behaviour has tests; bug fixes have a regression test
- [ ] Retrieval changes: `python -m src.pipeline.retrieval_eval` re-run and the README table updated, or marked as not re-run
- [ ] No API keys or credentials in code or fixtures
