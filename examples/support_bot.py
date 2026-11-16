"""A toy 'support bot' so the examples run offline.

In your project this would be a call to a real model, e.g.
    llm = promptunit.anthropic("claude-haiku-5-5")   # or openai(...), ollama(...), groq(...)
"""
import json

REFUND_POLICY = "Refunds are possible within 7 days of delivery."


def support_bot(prompt: str, **_kwargs) -> str:
    p = prompt.lower()
    if "json" in p and "order" in p:
        return json.dumps({"intent": "order_status", "order_id": "A-1042", "urgent": False})
    if "refund" in p:
        return f"Sure! {REFUND_POLICY} Please share your order ID and I'll start it."
    return "Namaste! How can I help you today?"
