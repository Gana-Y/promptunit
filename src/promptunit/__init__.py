"""promptunit: unit tests for your LLM prompts."""
from .expect import Expectation, PromptAssertionError, assert_pass_rate, expect, pass_rate
from .llm import (
    LLM,
    Anthropic,
    CacheMissError,
    Completion,
    FunctionLLM,
    LLMError,
    MockLLM,
    OpenAICompatible,
    anthropic,
    gemini,
    groq,
    ollama,
    openai,
    openrouter,
)

__version__ = "0.1.0"

__all__ = [
    "expect",
    "Expectation",
    "PromptAssertionError",
    "assert_pass_rate",
    "pass_rate",
    "LLM",
    "Completion",
    "OpenAICompatible",
    "Anthropic",
    "FunctionLLM",
    "MockLLM",
    "CacheMissError",
    "LLMError",
    "openai",
    "anthropic",
    "ollama",
    "groq",
    "openrouter",
    "gemini",
]
