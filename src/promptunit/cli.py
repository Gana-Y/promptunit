"""``promptunit`` command line."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__, config
from .cache import DiskCache

TEMPLATE = '''\
"""Example promptunit tests. Run with:  pytest"""
import os

import pytest

from promptunit import MockLLM, expect, assert_pass_rate, anthropic, openai, ollama


def make_llm():
    """Pick a model from env vars; fall back to a mock so the demo runs offline."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return anthropic("claude-haiku-5-5")
    if os.getenv("OPENAI_API_KEY"):
        return openai("gpt-4o-mini")
    if os.getenv("PROMPTUNIT_OLLAMA_MODEL"):
        return ollama(os.environ["PROMPTUNIT_OLLAMA_MODEL"])
    return MockLLM('{"sentiment": "positive", "confidence": 0.93}')


llm = make_llm()

PROMPT = (
    "Classify the sentiment of this review as JSON with keys 'sentiment' and 'confidence'. "
    "Return raw JSON only, no markdown.\\n\\nReview: The battery life is fantastic!"
)


def test_returns_raw_json():
    out = llm(PROMPT, temperature=0)
    expect(out).is_json().has_keys("sentiment", "confidence").no_refusal()


def test_sentiment_value_is_valid():
    data = expect(llm(PROMPT, temperature=0)).is_json().json
    assert data["sentiment"] in {"positive", "negative", "neutral"}


def test_json_is_reliable_across_samples():
    outputs = llm.sample(PROMPT, n=5, temperature=0.7)
    assert_pass_rate(outputs, lambda o: expect(o).is_json(), at_least=0.8)
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="promptunit", description="Unit tests for LLM prompts.")
    parser.add_argument("--version", action="version", version=f"promptunit {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="create tests/test_prompts.py with a working example")
    p_init.add_argument("directory", nargs="?", default="tests")

    p_cache = sub.add_parser("cache", help="inspect or clear the response cache")
    p_cache.add_argument("action", choices=["info", "clear"])
    p_cache.add_argument("--dir", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "init":
        os.makedirs(args.directory, exist_ok=True)
        path = os.path.join(args.directory, "test_prompts.py")
        if os.path.exists(path):
            print(f"{path} already exists; not overwriting.", file=sys.stderr)
            return 1
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE)
        print(f"Created {path}\nNow run:  pytest")
        return 0

    if args.cmd == "cache":
        cache = DiskCache(args.dir or config.get_cache_dir())
        if args.action == "info":
            count, size = cache.info()
            print(f"{cache.directory}: {count} cached responses, {size / 1024:.1f} KiB")
        else:
            print(f"Removed {cache.clear()} cached responses from {cache.directory}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
