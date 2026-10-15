"""Contract tests for the OpenAI / Anthropic adapters with fake SDK objects
(no network): request construction and response parsing."""

from types import SimpleNamespace as NS

from kyground.kb.replica_api import ReplicaAPI, ReplicaBackend
from kyground.llm.anthropic_client import AnthropicClient, _to_anthropic
from kyground.llm.openai_client import OpenAIClient
from kyground.systems.base import load_prompts, load_tools
from kyground.systems.conditions import ToolAgent
from test_kb import _kb


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        usage = NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=NS(cached_tokens=40),
                   completion_tokens_details=NS(reasoning_tokens=5))
        if kw.get("tools") and kw.get("tool_choice") == "auto":
            tc = NS(id="call_1", function=NS(name="search_local_products", arguments='{"query": "ΜΟΥΖΟΥΡΙ"}'))
            msg = NS(content=None, tool_calls=[tc])
            return NS(choices=[NS(message=msg, finish_reason="tool_calls")], usage=usage, model=kw["model"])
        msg = NS(content="Το παράγει ο Συνεταιρισμός Λιβαδίου.", tool_calls=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=usage, model=kw["model"])


def _fake_openai(model):
    c = OpenAIClient(model, cache_dir=None)
    fake = FakeCompletions()
    c._client = NS(chat=NS(completions=fake))
    return c, fake


def test_tool_agent_openai_contract():
    router, fr = _fake_openai("gpt-5-mini")
    answer, fa = _fake_openai("gpt-5-nano")
    s = _kb()
    agent = ToolAgent(answer, load_prompts(), router_llm=router, tools=load_tools(),
                      backend=ReplicaBackend(ReplicaAPI(s)), router_params={"reasoning_effort": "minimal"},
                      answer_params={"reasoning_effort": "low", "max_output_tokens": 4000})
    th = agent.new_thread("el")
    res = agent.ask(th, "Ποιος παράγει το ΜΟΥΖΟΥΡΙ 750ml;")
    assert res.error == ""
    r0 = fr.calls[0]
    assert r0["reasoning_effort"] == "minimal" and r0["tool_choice"] == "auto" and r0["parallel_tool_calls"] is True
    assert {t["function"]["name"] for t in r0["tools"]} >= {"search_kythera_knowledge", "get_seasonal_calendar"}
    a = fa.calls[-1]
    assert a["tool_choice"] == "none" and a["max_completion_tokens"] == 4000 and a["reasoning_effort"] == "low"
    roles = [m["role"] for m in a["messages"]]
    assert roles[-2:] == ["assistant", "tool"]
    assert a["messages"][-2]["tool_calls"][0]["function"]["name"] == "search_local_products"
    assert "MOUZOURI" in a["messages"][-1]["content"] or "ΜΟΥΖΟΥΡΙ" in a["messages"][-1]["content"]
    assert res.tool_calls[0]["ids"] == ["product:oil"]
    assert res.usage["gpt-5-nano"]["cached_input_tokens"] == 40
    assert th.history[-1]["content"].startswith("Το παράγει")


def test_anthropic_message_conversion():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "x", "arguments": "{\"a\": 1}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "{}"}]
    system, out = _to_anthropic(msgs)
    assert system == "S" and out[1]["content"][0]["type"] == "tool_use" and out[2]["content"][0]["type"] == "tool_result"


def test_anthropic_structured_output_request():
    c = AnthropicClient("claude-sonnet-5-5", cache_dir=None)
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(content=[NS(type="text", text='{"correctness": "correct", "unsupported_specifics": false, '
                                                '"rationale": "ok"}')],
                  usage=NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0), model=kw["model"],
                  stop_reason="end_turn")

    c._client = NS(messages=NS(create=create))
    schema = load_prompts()["judge"]["schema"]
    r = c.complete([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], response_schema=schema)
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in captured
    assert r.text.startswith("{") and r.error == ""
