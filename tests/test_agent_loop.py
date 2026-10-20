"""Tests for harness.run_agent using a scripted fake model - no API key or Ollama needed.

The fake client returns canned chat-completion messages, so these tests exercise everything
around the model for real: the tool-calling loop, routing to a real MCP server, the trace,
the step cap and error handling.
"""

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
    assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ["c1", "c2"]


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
