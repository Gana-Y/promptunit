"""Fluent assertions for LLM output.

    from promptunit import expect
    expect(output).is_json().has_keys("name", "age").not_contains("As an AI")
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Callable, Iterable, List, Optional, Sequence

from .llm import LLM, Completion


class PromptAssertionError(AssertionError):
    """Raised when an expectation about a model output fails."""


def _preview(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


# Unicode ranges used by ``in_script``. Latin is included for "answer in English" checks.
SCRIPT_RANGES = {
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "odia": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
}

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.S)

_REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) sorry, but\b",
    r"\bI can(?:'|no)t (?:help|assist|comply|do that|provide)\b",
    r"\bI(?:'m| am) (?:unable|not able) to\b",
    r"\bI (?:won't|will not) be able to\b",
    r"\bas an ai(?: language model)?\b",
]

_PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_in": r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)",
    "aadhaar": r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "card": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class Expectation:
    """Chainable assertions. Every method returns ``self`` or raises ``PromptAssertionError``."""

    def __init__(self, output: Any):
        self.output: str = str(output)
        self.meta: Optional[Completion] = output if isinstance(output, Completion) else None
        self._json: Any = None

    # -- plumbing ----------------------------------------------------------
    def _fail(self, message: str) -> "Expectation":
        __tracebackhide__ = True
        model = f" [{self.meta.model}]" if self.meta and self.meta.model else ""
        raise PromptAssertionError(
            f"{message}\n  output{model} ({len(self.output)} chars): \"{_preview(self.output)}\""
        )

    def satisfies(self, predicate: Callable[[str], bool], message: str = "custom check failed") -> "Expectation":
        """Escape hatch: any ``str -> bool`` function."""
        __tracebackhide__ = True
        if not predicate(self.output):
            self._fail(message)
        return self

    # -- content -----------------------------------------------------------
    def contains(self, *needles: str, case_sensitive: bool = False, any_of: bool = False) -> "Expectation":
        __tracebackhide__ = True
        hay = self.output if case_sensitive else self.output.lower()
        found = [n for n in needles if (n if case_sensitive else n.lower()) in hay]
        if any_of:
            if not found:
                self._fail(f"expected at least one of {list(needles)!r}")
        else:
            missing = [n for n in needles if n not in found]
            if missing:
                self._fail(f"expected output to contain {missing!r}")
        return self

    def not_contains(self, *needles: str, case_sensitive: bool = False) -> "Expectation":
        __tracebackhide__ = True
        hay = self.output if case_sensitive else self.output.lower()
        bad = [n for n in needles if (n if case_sensitive else n.lower()) in hay]
        if bad:
            self._fail(f"expected output NOT to contain {bad!r}")
        return self

    def matches(self, pattern: str, flags: int = re.IGNORECASE) -> "Expectation":
        __tracebackhide__ = True
        if not re.search(pattern, self.output, flags):
            self._fail(f"expected output to match regex {pattern!r}")
        return self

    def not_matches(self, pattern: str, flags: int = re.IGNORECASE) -> "Expectation":
        __tracebackhide__ = True
        if re.search(pattern, self.output, flags):
            self._fail(f"expected output NOT to match regex {pattern!r}")
        return self

    def starts_with(self, prefix: str) -> "Expectation":
        __tracebackhide__ = True
        if not self.output.lstrip().startswith(prefix):
            self._fail(f"expected output to start with {prefix!r}")
        return self

    def ends_with(self, suffix: str) -> "Expectation":
        __tracebackhide__ = True
        if not self.output.rstrip().endswith(suffix):
            self._fail(f"expected output to end with {suffix!r}")
        return self

    def equals(self, expected: str, *, normalize: bool = True) -> "Expectation":
        __tracebackhide__ = True
        a, b = self.output, expected
        if normalize:
            a, b = " ".join(a.split()).lower(), " ".join(b.split()).lower()
        if a != b:
            self._fail(f"expected output to equal {expected!r}")
        return self

    def one_of(self, options: Sequence[str], *, normalize: bool = True) -> "Expectation":
        __tracebackhide__ = True
        a = " ".join(self.output.split()).lower() if normalize else self.output
        opts = [" ".join(o.split()).lower() if normalize else o for o in options]
        if a not in opts:
            self._fail(f"expected output to be one of {list(options)!r}")
        return self

    # -- size --------------------------------------------------------------
    def max_chars(self, n: int) -> "Expectation":
        __tracebackhide__ = True
        if len(self.output) > n:
            self._fail(f"expected at most {n} characters, got {len(self.output)}")
        return self

    def min_chars(self, n: int) -> "Expectation":
        __tracebackhide__ = True
        if len(self.output) < n:
            self._fail(f"expected at least {n} characters, got {len(self.output)}")
        return self

    def max_words(self, n: int) -> "Expectation":
        __tracebackhide__ = True
        words = len(self.output.split())
        if words > n:
            self._fail(f"expected at most {n} words, got {words}")
        return self

    def min_words(self, n: int) -> "Expectation":
        __tracebackhide__ = True
        words = len(self.output.split())
        if words < n:
            self._fail(f"expected at least {n} words, got {words}")
        return self

    def max_lines(self, n: int) -> "Expectation":
        __tracebackhide__ = True
        lines = len([ln for ln in self.output.splitlines() if ln.strip()])
        if lines > n:
            self._fail(f"expected at most {n} non-empty lines, got {lines}")
        return self

    # -- structure ---------------------------------------------------------
    def is_json(self, *, allow_fences: bool = False) -> "Expectation":
        """Output must be valid JSON. Markdown fences fail unless ``allow_fences=True``."""
        __tracebackhide__ = True
        text = self.output.strip()
        if allow_fences:
            m = _FENCE.match(text)
            if m:
                text = m.group(1)
        try:
            self._json = json.loads(text)
        except json.JSONDecodeError as exc:
            hint = " (wrapped in a markdown code fence?)" if text.startswith("```") else ""
            self._fail(f"expected valid JSON: {exc.msg} at char {exc.pos}{hint}")
        return self

    @property
    def json(self) -> Any:
        """The parsed JSON (available after :meth:`is_json`)."""
        __tracebackhide__ = True
        if self._json is None:
            self.is_json()
        return self._json

    def has_keys(self, *keys: str) -> "Expectation":
        __tracebackhide__ = True
        data = self.json
        if not isinstance(data, dict):
            self._fail(f"expected a JSON object, got {type(data).__name__}")
        missing = [k for k in keys if k not in data]
        if missing:
            self._fail(f"JSON is missing keys {missing!r} (has {sorted(data)!r})")
        return self

    def json_matches(self, schema: dict) -> "Expectation":
        """Validate against a JSON Schema. Needs ``pip install promptunit[schema]``."""
        __tracebackhide__ = True
        try:
            import jsonschema
        except ImportError as exc:  # pragma: no cover
            raise ImportError("json_matches needs jsonschema: pip install 'promptunit[schema]'") from exc
        try:
            jsonschema.validate(self.json, schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
            self._fail(f"JSON does not match schema at {path}: {exc.message}")
        return self

    # -- safety / quality --------------------------------------------------
    def no_refusal(self) -> "Expectation":
        __tracebackhide__ = True
        for pat in _REFUSAL_PATTERNS:
            if re.search(pat, self.output, re.IGNORECASE):
                self._fail(f"output looks like a refusal / boilerplate (matched {pat!r})")
        return self

    def no_pii(self, kinds: Optional[Iterable[str]] = None) -> "Expectation":
        """Fail if output contains emails, Indian phone / Aadhaar / PAN numbers, SSNs or card numbers."""
        __tracebackhide__ = True
        wanted = list(kinds) if kinds else list(_PII_PATTERNS)
        unknown = [k for k in wanted if k not in _PII_PATTERNS]
        if unknown:
            raise ValueError(f"unknown PII kinds {unknown}; choose from {sorted(_PII_PATTERNS)}")
        for kind in wanted:
            for m in re.finditer(_PII_PATTERNS[kind], self.output):
                if kind == "card" and not _luhn_ok(m.group(0)):
                    continue
                self._fail(f"output contains {kind} data: {m.group(0)!r}")
        return self

    def in_script(self, script: str, min_ratio: float = 0.8) -> "Expectation":
        """Most letters must belong to ``script`` (e.g. 'kannada', 'devanagari', 'latin')."""
        __tracebackhide__ = True
        key = script.lower()
        if key not in SCRIPT_RANGES:
            raise ValueError(f"unknown script {script!r}; choose from {sorted(SCRIPT_RANGES)}")
        ranges = SCRIPT_RANGES[key]
        letters = [c for c in self.output if c.isalpha()]
        if not letters:
            self._fail(f"expected {script} text but output has no letters")
        hits = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
        ratio = hits / len(letters)
        if ratio < min_ratio:
            self._fail(f"expected >= {min_ratio:.0%} {script} letters, got {ratio:.0%}")
        return self

    def similar_to(self, reference: str, threshold: float = 0.8) -> "Expectation":
        """Cheap lexical similarity (difflib ratio, case/whitespace-insensitive). No embeddings needed."""
        __tracebackhide__ = True
        a = " ".join(self.output.split()).lower()
        b = " ".join(reference.split()).lower()
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if ratio < threshold:
            self._fail(f"similarity {ratio:.2f} < {threshold:.2f} to reference \"{_preview(reference, 80)}\"")
        return self

    def judged_by(self, judge: LLM, criteria: str) -> "Expectation":
        """LLM-as-judge: ask ``judge`` whether the output meets ``criteria``."""
        __tracebackhide__ = True
        from .judge import judge_output

        verdict = judge_output(judge, self.output, criteria)
        if not verdict.passed:
            self._fail(f"judge says FAIL for criteria {criteria!r}: {verdict.reason}")
        return self


def expect(output: Any) -> Expectation:
    """Start a chain of assertions on a model output."""
    return Expectation(output)


# ---------------------------------------------------------------------------
# Statistical checks: LLMs are non-deterministic, so test a *rate*, not one sample.
# ---------------------------------------------------------------------------
def _passes(check: Callable[[Any], Any], output: Any) -> bool:
    try:
        result = check(output)
    except AssertionError:
        return False
    return result is not False


def pass_rate(outputs: Sequence[Any], check: Callable[[Any], Any]) -> float:
    """Fraction of ``outputs`` for which ``check`` passes (returns truthy / raises nothing)."""
    if not outputs:
        raise ValueError("pass_rate needs at least one output")
    return sum(_passes(check, o) for o in outputs) / len(outputs)


def assert_pass_rate(outputs: Sequence[Any], check: Callable[[Any], Any], at_least: float = 0.9) -> float:
    """Assert that at least ``at_least`` of the outputs pass ``check``."""
    __tracebackhide__ = True
    rate = pass_rate(outputs, check)
    if rate < at_least:
        failures: List[str] = []
        for o in outputs:
            if not _passes(check, o):
                failures.append(_preview(str(o), 120))
            if len(failures) == 3:
                break
        shown = "\n".join(f'    - "{f}"' for f in failures)
        raise PromptAssertionError(
            f"pass rate {rate:.0%} < required {at_least:.0%} over {len(outputs)} samples\n"
            f"  example failures:\n{shown}"
        )
    return rate
