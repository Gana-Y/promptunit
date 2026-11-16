# Contributing to promptunit

Thanks for helping! Small PRs are the best PRs.

## Dev setup

```bash
git clone https://github.com/Gana-Y/promptunit.git
cd promptunit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Guidelines

- Add or update tests for every behaviour change.
- Keep the public API small and the docs honest: if something is a heuristic, say so.
- Open an issue first for big changes so we can agree on the approach.

By contributing you agree that your contribution is released under the MIT License.
