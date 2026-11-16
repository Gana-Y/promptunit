"""Run:  pytest examples -v"""
from promptunit import FunctionLLM, assert_pass_rate, expect

from support_bot import support_bot

bot = FunctionLLM(support_bot, name="support-bot")


def test_refund_answer_mentions_policy_and_is_short():
    out = bot("Can I get a refund?")
    expect(out).contains("7 days", "order ID").max_words(40).no_refusal()


def test_never_leaks_personal_data():
    expect(bot("Can I get a refund?")).no_pii()


def test_order_intent_is_raw_json_with_required_keys():
    out = bot("Return JSON for: where is my order A-1042?")
    expect(out).is_json().has_keys("intent", "order_id").no_refusal()


def test_json_is_reliable_across_samples():
    outputs = bot.sample("Return JSON for: where is my order?", n=5)
    assert_pass_rate(outputs, lambda o: expect(o).is_json(), at_least=1.0)


def test_greeting_is_in_latin_script_and_friendly():
    expect(bot("hello")).in_script("latin").contains("help")
