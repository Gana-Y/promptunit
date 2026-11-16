import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from promptunit import (
    Anthropic,
    CacheMissError,
    Completion,
    FunctionLLM,
    LLMError,
    MockLLM,
    OpenAICompatible,
    config,
    expect,
)
from promptunit.stats import stats


class _Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.path.endswith("/chat/completions"):
            if body["messages"][-1]["content"] == "boom":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"bad"}')
                return
            payload = {
                "choices": [{"message": {"content": "pong: " + body["messages"][-1]["content"]}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        elif self.path.endswith("/v1/messages"):
            payload = {
                "content": [{"type": "text", "text": "claude-pong"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    _Handler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", _Handler.requests
    srv.shutdown()


def test_completion_is_a_str_with_metadata():
    c = Completion("hi", model="m", input_tokens=1, output_tokens=2)
    assert c == "hi" and "h" in c and c.upper() == "HI"
    assert c.model == "m" and not c.cached
    assert Completion.from_dict(c.to_dict(), cached=True).cached


def test_openai_compatible_request_and_parsing(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1", api_key="sk-test")
    out = llm("ping", system="be brief", temperature=0)
    assert out == "pong: ping"
    assert (out.input_tokens, out.output_tokens, out.model) == (7, 3, "tiny")
    r = reqs[0]
    assert r["path"] == "/v1/chat/completions"
    assert {k.lower(): v for k, v in r["headers"].items()}["authorization"] == "Bearer sk-test"
    assert r["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert r["body"]["temperature"] == 0


def test_http_error_is_raised_without_retry(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1", cache=False)
    with pytest.raises(LLMError, match="HTTP 400"):
        llm("boom")
    assert len(reqs) == 1


def test_anthropic_adapter(server):
    url, reqs = server
    llm = Anthropic("some-model", api_key="k", base_url=url)
    out = llm("hello", system="sys")
    assert out == "claude-pong" and out.output_tokens == 2
    headers = {k.lower(): v for k, v in reqs[0]["headers"].items()}
    assert headers["x-api-key"] == "k" and headers["anthropic-version"]
    assert reqs[0]["body"]["system"] == "sys"
    assert reqs[0]["body"]["max_tokens"] == 1024


def test_cache_auto_mode_second_call_is_free(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1")
    a = llm("ping")
    b = llm("ping")
    assert a == b and not a.cached and b.cached
    assert len(reqs) == 1
    assert stats.calls == 2 and stats.cache_hits == 1


def test_cache_key_depends_on_prompt_params_and_sample(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1")
    llm("a")
    llm("b")
    llm("a", temperature=0.5)
    llm("a", sample=1)
    assert len(reqs) == 4
    llm("a")  # cached
    assert len(reqs) == 4


def test_replay_mode_raises_on_miss_and_hits_when_recorded(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1")
    config.set_mode("replay")
    with pytest.raises(CacheMissError, match="No recorded response"):
        llm("never seen")
    assert reqs == []
    config.set_mode("record")
    llm("seen")
    config.set_mode("replay")
    assert llm("seen") == "pong: seen"
    assert len(reqs) == 1


def test_live_mode_bypasses_cache(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1")
    llm("x")
    config.set_mode("live")
    llm("x")
    assert len(reqs) == 2


def test_sample_returns_n_distinct_cache_entries(server):
    url, reqs = server
    llm = OpenAICompatible("tiny", base_url=url + "/v1")
    outs = llm.sample("x", n=3)
    assert len(outs) == 3 and len(reqs) == 3
    llm.sample("x", n=3)
    assert len(reqs) == 3


def test_invalid_mode():
    with pytest.raises(ValueError):
        config.set_mode("nope")


def test_function_llm_wraps_your_app():
    llm = FunctionLLM(lambda p, **kw: p.upper(), name="shout")
    out = llm("hey")
    assert out == "HEY" and out.model == "shout"
    expect(out).contains("hey")


def test_mock_llm_variants():
    assert MockLLM("fixed")("x") == "fixed"
    seq = MockLLM(["a", "b"])
    assert [seq("x"), seq("x"), seq("x")] == ["a", "b", "a"]
    assert MockLLM({"weather": "sunny"})("what is the weather") == "sunny"
    with pytest.raises(LLMError):
        MockLLM({"weather": "sunny"})("stocks")
    assert MockLLM(lambda p: p[::-1])("abc") == "cba"
    m = MockLLM("ok")
    m("q", system="s", temperature=0)
    assert m.calls[0]["system"] == "s"


def test_cost_estimate():
    llm = MockLLM("x", price_per_1m=(1.0, 2.0))
    c = Completion("x", input_tokens=1_000_000, output_tokens=500_000)
    assert llm._cost(c) == pytest.approx(2.0)


def test_judge_with_mock():
    judge = MockLLM('{"pass": true, "reason": "polite"}')
    expect("Thank you!").judged_by(judge, "is polite")
    bad = MockLLM('Sure! {"pass": false, "reason": "rude"}')
    with pytest.raises(AssertionError, match="rude"):
        expect("shut up").judged_by(bad, "is polite")
    garbled = MockLLM("I think so")
    with pytest.raises(AssertionError, match="unparseable"):
        expect("x").judged_by(garbled, "anything")
    assert "CRITERIA" in judge.calls[0]["prompt"] and judge.calls[0]["params"] == {"temperature": 0}
