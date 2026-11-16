import pytest

from promptunit import PromptAssertionError, assert_pass_rate, expect, pass_rate


def fails(fn):
    with pytest.raises(PromptAssertionError) as info:
        fn()
    return str(info.value)


def test_contains_and_not_contains():
    expect("Your refund will arrive in 5 days").contains("refund", "5 days")
    expect("Hello").contains("nope", "hello", any_of=True)
    assert "refund" in fails(lambda: expect("Sorry").contains("refund"))
    fails(lambda: expect("As an AI model").not_contains("as an ai"))
    expect("As an AI model").not_contains("as an ai", case_sensitive=True)


def test_matches_start_end_equals_one_of():
    expect("Order #12345 shipped").matches(r"#\d{5}")
    fails(lambda: expect("no number").matches(r"\d+"))
    fails(lambda: expect("Order #1").not_matches(r"#\d"))
    expect("  Hello world").starts_with("Hello")
    expect("done.  ").ends_with("done.")
    expect("  YES ").equals("yes")
    fails(lambda: expect("yes").equals("no"))
    expect("Positive").one_of(["positive", "negative"])
    fails(lambda: expect("maybe").one_of(["positive", "negative"]))


def test_size_limits():
    expect("one two three").max_words(3).min_words(3).max_chars(13).min_chars(13)
    fails(lambda: expect("one two three four").max_words(3))
    fails(lambda: expect("a").min_chars(5))
    fails(lambda: expect("a\nb\nc").max_lines(2))
    expect("a\n\nb").max_lines(2)


def test_json_checks():
    e = expect('{"name": "Asha", "age": 31}').is_json().has_keys("name", "age")
    assert e.json["age"] == 31
    msg = fails(lambda: expect('```json\n{"a": 1}\n```').is_json())
    assert "code fence" in msg
    expect('```json\n{"a": 1}\n```').is_json(allow_fences=True).has_keys("a")
    fails(lambda: expect("not json").is_json())
    fails(lambda: expect('{"a": 1}').has_keys("b"))
    fails(lambda: expect("[1, 2]").has_keys("a"))


def test_json_schema():
    pytest.importorskip("jsonschema")
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
    expect('{"age": 3}').json_matches(schema)
    msg = fails(lambda: expect('{"age": "x"}').json_matches(schema))
    assert "age" in msg


def test_refusal_detection():
    fails(lambda: expect("I'm sorry, but I can't help with that.").no_refusal())
    fails(lambda: expect("As an AI language model, I think...").no_refusal())
    expect("Sure, here is the answer.").no_refusal()


def test_pii_detection():
    expect("Contact support please").no_pii()
    fails(lambda: expect("mail me at a.b@example.com").no_pii())
    fails(lambda: expect("call +91 98765 43210").no_pii(["phone_in"]))
    fails(lambda: expect("call 9876543210").no_pii(["phone_in"]))
    fails(lambda: expect("PAN is ABCDE1234F").no_pii(["pan"]))
    fails(lambda: expect("aadhaar 2345 6789 0123").no_pii(["aadhaar"]))
    fails(lambda: expect("card 4111 1111 1111 1111").no_pii(["card"]))
    expect("order 1234 5678 9012 3456").no_pii(["card"])  # fails Luhn -> not a card
    with pytest.raises(ValueError):
        expect("x").no_pii(["bogus"])


def test_in_script():
    expect("ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ").in_script("kannada")
    expect("नमस्ते दोस्तों").in_script("devanagari")
    expect("Hello there").in_script("latin")
    fails(lambda: expect("Hello there").in_script("kannada"))
    fails(lambda: expect("12345").in_script("kannada"))
    with pytest.raises(ValueError):
        expect("x").in_script("klingon")


def test_similar_to():
    expect("The quick brown fox").similar_to("the quick   brown fox", 0.95)
    fails(lambda: expect("totally different").similar_to("The quick brown fox", 0.8))


def test_satisfies_and_message_contains_output():
    msg = fails(lambda: expect("abc").satisfies(lambda s: s == "x", "must be x"))
    assert "must be x" in msg and '"abc"' in msg


def test_multiline_output_is_escaped_once():
    msg = fails(lambda: expect("line1\nline2").contains("zzz"))
    assert '"line1\\nline2"' in msg  # single backslash-n, not doubled


def test_pass_rate():
    outs = ['{"a":1}', '{"a":2}', "oops", '{"a":3}']
    check = lambda o: expect(o).is_json()
    assert pass_rate(outs, check) == 0.75
    assert assert_pass_rate(outs, check, at_least=0.7) == 0.75
    msg = fails(lambda: assert_pass_rate(outs, check, at_least=0.9))
    assert "75%" in msg and "oops" in msg
    with pytest.raises(ValueError):
        pass_rate([], check)
