"""LLM adapters. Zero dependencies: everything uses the standard library."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import config
from .cache import DiskCache, make_key
from .stats import stats


class CacheMissError(RuntimeError):
    """Raised in ``replay`` mode when a call has not been recorded yet."""


class LLMError(RuntimeError):
    """Raised when a provider returns an error."""


class Completion(str):
    """A model response.

    It *is* a ``str`` (so ``"refund" in out`` just works) but also carries
    metadata: ``model``, ``latency_s``, ``input_tokens``, ``output_tokens``, ``cached``.
    """

    model: str
    latency_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached: bool

    def __new__(
        cls,
        text: str,
        *,
        model: str = "",
        latency_s: float = 0.0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached: bool = False,
    ) -> "Completion":
        obj = super().__new__(cls, text)
        obj.model = model
        obj.latency_s = latency_s
        obj.input_tokens = input_tokens
        obj.output_tokens = output_tokens
        obj.cached = cached
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": str(self),
            "model": self.model,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], cached: bool = False) -> "Completion":
        return cls(
            data["text"],
            model=data.get("model", ""),
            latency_s=data.get("latency_s", 0.0),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            cached=cached,
        )


class LLM:
    """Base class. Subclasses implement :meth:`_complete`."""

    #: identifies this model/endpoint in the cache key
    cache_id: str = "llm"
    #: whether responses are cached on disk
    use_cache: bool = True

    def __init__(
        self,
        *,
        cache: Optional[bool] = None,
        price_per_1m: Optional[Tuple[float, float]] = None,
    ):
        if cache is not None:
            self.use_cache = cache
        self.price_per_1m = price_per_1m

    # -- to implement ------------------------------------------------------
    def _complete(self, prompt: str, system: Optional[str], params: Dict[str, Any]) -> Completion:
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def __call__(self, prompt: str, *, system: Optional[str] = None, sample: int = 0, **params: Any) -> Completion:
        return self.call(prompt, system=system, sample=sample, **params)

    def call(self, prompt: str, *, system: Optional[str] = None, sample: int = 0, **params: Any) -> Completion:
        mode = config.get_mode()
        cache = DiskCache(config.get_cache_dir()) if self.use_cache else None
        key = make_key(
            {"id": self.cache_id, "prompt": prompt, "system": system, "params": params, "sample": sample}
        )
        if cache is not None and mode in ("auto", "replay"):
            hit = cache.get(key)
            if hit is not None:
                completion = Completion.from_dict(hit, cached=True)
                stats.record(completion)
                return completion
            if mode == "replay":
                raise CacheMissError(
                    "No recorded response for this call (mode=replay).\n"
                    f"  model : {self.cache_id}\n"
                    f"  prompt: {prompt[:120]!r}\n"
                    "Run once with --promptunit-mode=auto (or record) and commit the cache directory."
                )
        completion = self._complete(prompt, system, params)
        if cache is not None and mode in ("auto", "record"):
            cache.put(key, completion.to_dict())
        stats.record(completion, self._cost(completion))
        return completion

    def sample(self, prompt: str, n: int = 5, **kwargs: Any) -> List[Completion]:
        """Call the model ``n`` times (each sample is cached separately)."""
        return [self.call(prompt, sample=i, **kwargs) for i in range(n)]

    def _cost(self, completion: Completion) -> float:
        if not self.price_per_1m:
            return 0.0
        pin, pout = self.price_per_1m
        return ((completion.input_tokens or 0) * pin + (completion.output_tokens or 0) * pout) / 1_000_000


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------
def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float, retries: int = 3) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    delay = 1.0
    last: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last = LLMError(f"HTTP {exc.code} from {url}: {detail}")
            if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last from None
        except (urllib.error.URLError, TimeoutError) as exc:
            last = LLMError(f"Network error calling {url}: {exc}")
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    assert last is not None
    raise last


class OpenAICompatible(LLM):
    """Any server that speaks the OpenAI ``/chat/completions`` protocol.

    Works with OpenAI, Ollama, Groq, OpenRouter, Gemini's OpenAI endpoint,
    vLLM, LM Studio, llama.cpp server and many others.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.cache_id = f"openai-compatible|{self.base_url}|{model}"

    def _complete(self, prompt, system, params):
        key = self.api_key or os.environ.get(self.api_key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        data = _post_json(
            f"{self.base_url}/chat/completions",
            headers,
            {"model": self.model, "messages": messages, **params},
            self.timeout,
        )
        latency = time.perf_counter() - t0
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from None
        usage = data.get("usage") or {}
        return Completion(
            text,
            model=self.model,
            latency_s=latency,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


class Anthropic(LLM):
    """Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_id = f"anthropic|{self.base_url}|{model}"

    def _complete(self, prompt, system, params):
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(params)
        if system:
            body["system"] = system
        t0 = time.perf_counter()
        data = _post_json(
            f"{self.base_url}/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            body,
            self.timeout,
        )
        latency = time.perf_counter() - t0
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from None
        usage = data.get("usage") or {}
        return Completion(
            text,
            model=self.model,
            latency_s=latency,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


class FunctionLLM(LLM):
    """Wrap *your own* function (a RAG chain, an agent, a prompt template...).

    ``fn(prompt, **kwargs) -> str``. Not cached by default.
    """

    use_cache = False

    def __init__(self, fn: Callable[..., str], *, name: str = "function", **kwargs: Any):
        super().__init__(**kwargs)
        self.fn = fn
        self.cache_id = f"function|{name}"
        self.name = name

    def _complete(self, prompt, system, params):
        t0 = time.perf_counter()
        kwargs = dict(params)
        if system is not None:
            kwargs["system"] = system
        text = self.fn(prompt, **kwargs)
        return Completion(str(text), model=self.name, latency_s=time.perf_counter() - t0)


class MockLLM(LLM):
    """Deterministic fake model for unit-testing your tests and for offline CI.

    ``responses`` can be a list (returned in order, cycling), a dict mapping a
    substring of the prompt to a response, or a callable ``prompt -> str``.
    """

    use_cache = False

    def __init__(self, responses: Union[Sequence[str], Dict[str, str], Callable[[str], str], str] = "OK", **kwargs: Any):
        super().__init__(**kwargs)
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []
        self._i = 0
        self.cache_id = "mock"

    def _complete(self, prompt, system, params):
        self.calls.append({"prompt": prompt, "system": system, "params": params})
        r = self.responses
        if callable(r):
            text = r(prompt)
        elif isinstance(r, str):
            text = r
        elif isinstance(r, dict):
            text = next((v for k, v in r.items() if k in prompt), None)
            if text is None:
                raise LLMError(f"MockLLM has no response matching prompt {prompt[:80]!r}")
        else:
            text = r[self._i % len(r)]
            self._i += 1
        return Completion(text, model="mock")


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
def openai(model: str, **kw: Any) -> OpenAICompatible:
    return OpenAICompatible(model, **kw)


def anthropic(model: str, **kw: Any) -> Anthropic:
    return Anthropic(model, **kw)


def ollama(model: str, host: str = "http://localhost:11434", **kw: Any) -> OpenAICompatible:
    """Local models via Ollama. No API key, no cost."""
    return OpenAICompatible(model, base_url=f"{host.rstrip('/')}/v1", api_key_env="OLLAMA_API_KEY", **kw)


def groq(model: str, **kw: Any) -> OpenAICompatible:
    return OpenAICompatible(model, base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY", **kw)


def openrouter(model: str, **kw: Any) -> OpenAICompatible:
    return OpenAICompatible(model, base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY", **kw)


def gemini(model: str, **kw: Any) -> OpenAICompatible:
    return OpenAICompatible(
        model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        **kw,
    )
