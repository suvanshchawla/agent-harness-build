"""Tests for harness.run_agent using a scripted fake model - no API key or Ollama needed.

The fake client returns canned chat-completion messages, so these tests exercise everything
around the model for real: the tool-calling loop, routing to a real MCP server, the trace,
the step cap and error handling.
"""

import pytest
from fakes import FakeClient, reply, tool_call

from agent_harness_build import harness


async def test_plain_answer_uses_no_tools(connected_harness):
    fake = FakeClient(reply("Hello!"))
    history: list = []
    trace: list = []
    async with connected_harness():
        answer = await harness.run_agent("hi", history=history, trace=trace, client=fake, verbose=False)
    assert answer == "Hello!"
    assert trace == []
    assert [m["role"] for m in history] == ["user", "assistant"]


async def test_tool_call_is_routed_to_the_real_server_and_result_fed_back(connected_harness):
    fake = FakeClient(
        reply(tool_calls=[tool_call("workspace__write_file", {"filename": "todo.md", "content": "milk"})]),
        reply("Saved it."),
    )
    trace: list = []
    async with connected_harness() as (workspace, _):
        answer = await harness.run_agent("add milk", history=[], trace=trace, client=fake, verbose=False)

        assert (workspace / "todo.md").read_text() == "milk"  # the real server did the work
    assert answer == "Saved it."
    assert trace == [
        {
            "tool": "workspace__write_file",
            "args": {"filename": "todo.md", "content": "milk"},
            "result": "wrote todo.md",
        }
    ]
    # the second model call must include the tool's result, tied to the call id
    tool_messages = [m for m in fake.requests[1] if isinstance(m, dict) and m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "tool_call_id": "call_1", "content": "wrote todo.md"}]


async def test_several_tool_calls_in_one_message_run_in_order(connected_harness):
    fake = FakeClient(
        reply(
            tool_calls=[
                tool_call("workspace__write_file", {"filename": "a.txt", "content": "1"}, "c1"),
                tool_call("workspace__write_file", {"filename": "b.txt", "content": "2"}, "c2"),
                tool_call("workspace__list_files", {}, "c3"),
            ]
        ),
        reply("done"),
    )
    trace: list = []
    async with connected_harness():
        await harness.run_agent("make files", history=[], trace=trace, client=fake, verbose=False)
    assert [t["tool"] for t in trace] == [
        "workspace__write_file",
        "workspace__write_file",
        "workspace__list_files",
    ]
    assert trace[2]["result"] == "a.txt\nb.txt"
    ids = [m["tool_call_id"] for m in fake.requests[1] if isinstance(m, dict) and m["role"] == "tool"]
    assert ids == ["c1", "c2", "c3"]


async def test_unknown_tool_error_goes_back_to_the_model(connected_harness):
    fake = FakeClient(reply(tool_calls=[tool_call("nope__missing", {})]), reply("Sorry."))
    trace: list = []
    async with connected_harness():
        answer = await harness.run_agent("x", history=[], trace=trace, client=fake, verbose=False)
    assert answer == "Sorry."
    assert trace[0]["result"] == "error: unknown tool nope__missing"


async def test_server_side_errors_are_returned_as_tool_results(connected_harness):
    fake = FakeClient(
        reply(tool_calls=[tool_call("workspace__read_file", {"filename": "../../etc/passwd"})]), reply("No.")
    )
    trace: list = []
    async with connected_harness():
        await harness.run_agent("x", history=[], trace=trace, client=fake, verbose=False)
    assert "outside the workspace" in trace[0]["result"]


async def test_loop_stops_at_the_step_cap(connected_harness, monkeypatch):
    monkeypatch.setattr(harness, "MAX_STEPS", 3)
    fake = FakeClient(reply(tool_calls=[tool_call("workspace__list_files", {})]))  # never stops
    trace: list = []
    async with connected_harness():
        answer = await harness.run_agent("loop", history=[], trace=trace, client=fake, verbose=False)
    assert answer == "(stopped after 3 steps without a final answer)"
    assert len(trace) == 3
    assert len(fake.requests) == 3


async def test_history_accumulates_across_turns(connected_harness):
    fake = FakeClient(reply("one"), reply("two"))
    history: list = [{"role": "system", "content": "sys"}]
    async with connected_harness():
        await harness.run_agent("first", history=history, client=fake, verbose=False)
        await harness.run_agent("second", history=history, client=fake, verbose=False)
    assert [m["content"] for m in history] == ["sys", "first", "one", "second", "two"]
    assert len(fake.requests[1]) == 4  # system + first + one + second


async def test_streamed_answer_arrives_in_pieces_and_matches_the_full_text(connected_harness):
    fake = FakeClient(reply("Hello there, friend!"))
    pieces: list[str] = []
    history: list = []
    async with connected_harness():
        answer = await harness.run_agent(
            "hi", history=history, client=fake, verbose=False, on_text=pieces.append
        )
    assert answer == "Hello there, friend!"
    assert len(pieces) > 1
    assert "".join(pieces) == answer
    assert history[-1] == {"role": "assistant", "content": "Hello there, friend!"}


async def test_streamed_tool_call_fragments_are_reassembled(connected_harness):
    fake = FakeClient(
        reply(
            tool_calls=[
                tool_call("workspace__write_file", {"filename": "a.txt", "content": "one"}, "c1"),
                tool_call("workspace__write_file", {"filename": "b.txt", "content": "two"}, "c2"),
            ]
        ),
        reply("Done."),
    )
    trace: list = []
    history: list = []
    async with connected_harness() as (workspace, _):
        answer = await harness.run_agent(
            "write", history=history, trace=trace, client=fake, verbose=False, on_text=lambda _: None
        )
        assert (workspace / "a.txt").read_text() == "one"
        assert (workspace / "b.txt").read_text() == "two"
    assert answer == "Done."
    assert [t["args"]["filename"] for t in trace] == ["a.txt", "b.txt"]
    assistant = history[1]
    assert [c["id"] for c in assistant["tool_calls"]] == ["c1", "c2"]
    tool_msgs = [m for m in history if isinstance(m, dict) and m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


async def test_empty_reply_never_puts_null_content_in_the_history(connected_harness):
    # Ollama rejects `content: null` on the next request, which crashed the session
    fake = FakeClient(reply(""))
    history: list = []
    async with connected_harness():
        await harness.run_agent("hi", history=history, client=fake, verbose=False, on_text=lambda _: None)
        await harness.run_agent("again", history=history, client=fake, verbose=False, on_text=lambda _: None)
    assert history[1] == {"role": "assistant", "content": harness.EMPTY_NOTICE}
    assert all(m.get("content") is not None for m in fake.requests[-1])


async def test_empty_reply_is_retried(connected_harness):
    fake = FakeClient(reply(""), reply("Here you go."))
    history: list = []
    async with connected_harness():
        answer = await harness.run_agent(
            "hi", history=history, client=fake, verbose=False, on_text=lambda _: None
        )
    assert answer == "Here you go."
    assert len(fake.requests) == 2
    assert fake.requests[1] == fake.requests[0]  # same conversation, nothing added for the retry
    assert history[-1] == {"role": "assistant", "content": "Here you go."}


async def test_still_empty_after_retry_falls_back_to_the_reasoning(connected_harness):
    fake = FakeClient(reply("", reasoning="The repo is a Flask API."))  # repeats forever
    pieces: list[str] = []
    history: list = []
    async with connected_harness():
        answer = await harness.run_agent(
            "hi", history=history, client=fake, verbose=False, on_text=pieces.append
        )
    assert answer == "The repo is a Flask API."
    assert "".join(pieces) == answer  # the user sees it, not just the return value
    assert len(fake.requests) == harness.EMPTY_RETRIES + 1
    assert history[-1]["content"] == answer


async def test_empty_with_no_reasoning_says_so_instead_of_returning_nothing(connected_harness):
    fake = FakeClient(reply(""))
    async with connected_harness():
        streamed = await harness.run_agent(
            "hi", history=[], client=fake, verbose=False, on_text=lambda _: None
        )
        plain = await harness.run_agent("hi", history=[], client=FakeClient(reply(None)), verbose=False)
    assert streamed == plain == harness.EMPTY_NOTICE


async def test_a_tool_call_with_no_text_is_not_treated_as_empty(connected_harness):
    fake = FakeClient(reply(tool_calls=[tool_call("workspace__list_files", {})]), reply("ok"))
    async with connected_harness():
        answer = await harness.run_agent("ls", history=[], client=fake, verbose=False, on_text=lambda _: None)
    assert answer == "ok"
    assert len(fake.requests) == 2  # no retry burned on the tool-call step


async def test_a_failed_model_call_rolls_the_turn_back_and_the_session_carries_on(connected_harness):
    fake = FakeClient(reply("one"), RuntimeError("400: bad request"), reply("three"))
    history: list = [{"role": "system", "content": "sys"}]
    async with connected_harness():
        await harness.run_agent("first", history=history, client=fake, verbose=False)
        before = list(history)
        with pytest.raises(RuntimeError, match="400"):
            await harness.run_agent("second", history=history, client=fake, verbose=False)
        assert history == before  # the failed message is gone, the earlier turn is intact
        await harness.run_agent("third", history=history, client=fake, verbose=False)
    assert [m["content"] for m in history] == ["sys", "first", "one", "third", "three"]


async def test_a_failure_after_tools_ran_removes_the_whole_turn(connected_harness):
    fake = FakeClient(
        reply(tool_calls=[tool_call("workspace__list_files", {})]), ConnectionError("connection dropped")
    )
    history: list = []
    async with connected_harness():
        with pytest.raises(ConnectionError):
            await harness.run_agent("ls", history=history, client=fake, verbose=False)
    assert history == []  # no orphaned tool call or tool result left behind


async def test_a_stream_that_breaks_midway_rolls_back_too(connected_harness):
    broken = reply("partial answer that never finishes")
    broken.fail = ConnectionError("stream cut")
    history: list = []
    async with connected_harness():
        with pytest.raises(ConnectionError):
            await harness.run_agent(
                "hi", history=history, client=FakeClient(broken), verbose=False, on_text=lambda _: None
            )
    assert history == []


async def test_malformed_tool_arguments_go_back_to_the_model_as_an_error(connected_harness):
    bad = tool_call("workspace__write_file", {}, "c1")
    bad.function.arguments = '{"filename": "a.txt", "content": "unfinished'  # cut off mid-string
    fake = FakeClient(
        reply(tool_calls=[bad, tool_call("workspace__list_files", {}, "c2")]),
        reply("Let me retry."),
    )
    trace: list = []
    history: list = []
    async with connected_harness() as (workspace, _):
        answer = await harness.run_agent("write", history=history, trace=trace, client=fake, verbose=False)
        assert not (workspace / "a.txt").exists()  # nothing half-written
    assert answer == "Let me retry."
    assert "not valid JSON" in trace[0]["result"]
    assert trace[1]["tool"] == "workspace__list_files"  # the sibling call still ran
    tool_msgs = [m for m in history if isinstance(m, dict) and m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


async def test_tool_arguments_that_are_not_an_object_are_rejected(connected_harness):
    odd = tool_call("workspace__list_files", {}, "c1")
    odd.function.arguments = "[1, 2]"
    fake = FakeClient(reply(tool_calls=[odd]), reply("ok"))
    trace: list = []
    async with connected_harness():
        await harness.run_agent("x", history=[], trace=trace, client=fake, verbose=False)
    assert "must be a JSON object" in trace[0]["result"]


async def test_chat_reports_a_failed_turn_and_keeps_the_session_alive(monkeypatch, capsys):
    async def boom(*_, **__):
        raise RuntimeError("400: invalid message content type")

    monkeypatch.setattr(harness, "run_agent", boom)
    await harness.chat("hello")  # must not raise
    out = capsys.readouterr().out
    assert "RuntimeError: 400: invalid message content type" in out
    assert "dropped" in out


def test_short_results_pass_through_untouched():
    assert harness.truncate_result("hello", limit=10) == "hello"
    assert harness.truncate_result("x" * 10, limit=10) == "x" * 10  # exactly at the limit


def test_long_results_are_cut_and_say_how_much_is_missing():
    cut = harness.truncate_result("a" * 1000 + "b" * 2345, limit=1000)
    assert cut.startswith("a" * 1000)
    assert "b" not in cut.split("[truncated")[0]
    assert "[truncated: 2,345 more characters not shown" in cut


async def test_the_model_only_sees_the_truncated_result(connected_harness, monkeypatch):
    monkeypatch.setattr(harness, "MAX_TOOL_CHARS", 100)
    fake = FakeClient(
        reply(tool_calls=[tool_call("workspace__read_file", {"filename": "big.txt"})]), reply("ok")
    )
    trace: list = []
    async with connected_harness() as (workspace, _):
        workspace.mkdir(exist_ok=True)
        (workspace / "big.txt").write_text("z" * 5000)
        await harness.run_agent("read it", history=[], trace=trace, client=fake, verbose=False)
    sent = [m for m in fake.requests[1] if isinstance(m, dict) and m["role"] == "tool"][0]["content"]
    assert sent == trace[0]["result"]  # the trace records what the model actually saw
    assert sent.startswith("z" * 100) and "[truncated: 4,900 more characters" in sent
    assert len(sent) < 300


async def test_short_error_results_are_not_mangled(connected_harness):
    async with connected_harness():
        result = await harness.call_tool("nope__missing", {})
    assert result == "error: unknown tool nope__missing"


async def test_usage_sums_every_model_call_in_the_turn(connected_harness):
    fake = FakeClient(
        reply(tool_calls=[tool_call("workspace__list_files", {})], tokens=(100, 10)),
        reply("done", tokens=(150, 20)),
    )
    usage: dict = {}
    async with connected_harness():
        await harness.run_agent("ls", history=[], client=fake, verbose=False, usage=usage)
    assert usage == {"model_calls": 2, "prompt_tokens": 250, "completion_tokens": 30}


async def test_usage_counts_the_retry_after_an_empty_reply(connected_harness):
    fake = FakeClient(reply("", tokens=(100, 0)), reply("ok", tokens=(100, 5)))
    usage: dict = {}
    async with connected_harness():
        await harness.run_agent("hi", history=[], client=fake, verbose=False, usage=usage)
    assert usage == {"model_calls": 2, "prompt_tokens": 200, "completion_tokens": 5}


async def test_usage_without_reported_tokens_counts_calls_but_invents_no_tokens(connected_harness):
    usage: dict = {}
    async with connected_harness():
        await harness.run_agent("hi", history=[], client=FakeClient(reply("ok")), verbose=False, usage=usage)
    assert usage == {"model_calls": 1}


async def test_streamed_usage_is_requested_only_when_a_usage_dict_is_given(connected_harness):
    asked, not_asked = FakeClient(reply("ok", tokens=(7, 3))), FakeClient(reply("ok", tokens=(7, 3)))
    usage: dict = {}
    async with connected_harness():
        await harness.run_agent(
            "hi", history=[], client=asked, verbose=False, on_text=lambda _: None, usage=usage
        )
        await harness.run_agent("hi", history=[], client=not_asked, verbose=False, on_text=lambda _: None)
    assert usage == {"model_calls": 1, "prompt_tokens": 7, "completion_tokens": 3}
    assert asked.stream_options == [{"include_usage": True}]
    assert not_asked.stream_options == [None]  # providers that reject the option are never sent it
