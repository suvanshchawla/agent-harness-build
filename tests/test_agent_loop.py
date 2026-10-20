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
