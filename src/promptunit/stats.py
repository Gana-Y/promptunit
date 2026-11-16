"""Run statistics shown in the pytest terminal summary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Stats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    by_model: Dict[str, int] = field(default_factory=dict)

    def record(self, completion, cost: float = 0.0) -> None:
        self.calls += 1
        if completion.cached:
            self.cache_hits += 1
        else:
            self.latency_s += completion.latency_s
        self.input_tokens += completion.input_tokens or 0
        self.output_tokens += completion.output_tokens or 0
        self.cost_usd += cost
        name = completion.model or "unknown"
        self.by_model[name] = self.by_model.get(name, 0) + 1

    def reset(self) -> None:
        self.calls = self.cache_hits = self.input_tokens = self.output_tokens = 0
        self.latency_s = self.cost_usd = 0.0
        self.by_model = {}


stats = Stats()
