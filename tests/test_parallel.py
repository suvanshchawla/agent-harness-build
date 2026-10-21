"""Parallel tool calls: neighbouring read-only calls overlap, everything else stays in order.

The slow test server logs when each call starts and ends, so overlap is checked from the log,
not from wall-clock timing.
"""

import pytest
from fakes import FakeClient, reply, tool_call

from agent_harness_build import harness


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(harness, "approval_mode", "writes")
    monkeypatch.setattr(harness, "always_allowed", set())


def calls(*specs):
    """specs: (tool, label[, delay]) -> tool_call objects with ids c0, c1, ..."""
    out = []
    for i, (tool, label, *delay) in enumerate(specs):
        args = {"label": label, **({"delay": delay[0]} if delay else {})}
        out.append(tool_call(f"slow__{tool}", args, f"c{i}"))
    return out


async def run(script_calls, **kwargs):
    trace: list = []
    history: list = []
    fake = FakeClient(reply(tool_calls=script_calls), reply("done"))
    await harness.run_agent("go", history=history, trace=trace, client=fake, verbose=False, **kwargs)
    return trace, history


def overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


async def test_neighbouring_reads_run_at_the_same_time(slow_harness):
    async with slow_harness() as intervals:
        await run(calls(("read", "a"), ("read", "b"), ("read", "c")))
        t = intervals()
    assert overlap(t["a"], t["b"]) and overlap(t["b"], t["c"]) and overlap(t["a"], t["c"])
    assert max(s for s, _ in t.values()) < min(e for _, e in t.values())  # all three in flight together


async def test_parallel_false_runs_them_one_after_another(slow_harness):
    async with slow_harness() as intervals:
        await run(calls(("read", "a"), ("read", "b"), ("read", "c")), parallel=False)
        t = intervals()
    assert t["a"][1] <= t["b"][0] and t["b"][1] <= t["c"][0]


async def test_parallel_is_actually_faster(slow_harness):
    async with slow_harness() as intervals:
        await run(calls(("read", "a", 0.4), ("read", "b", 0.4), ("read", "c", 0.4)))
        t = intervals()
    elapsed = max(e for _, e in t.values()) - min(s for s, _ in t.values())
    assert elapsed < 0.4 * 2  # three sequential calls would need 1.2s


async def test_a_write_is_never_reordered_or_overlapped(slow_harness):
    async with slow_harness() as intervals:
        await run(calls(("read", "r1"), ("read", "r2"), ("write", "w"), ("read", "r3")))
        t = intervals()
    assert overlap(t["r1"], t["r2"])  # the two reads before the write share a batch
    assert t["w"][0] >= max(t["r1"][1], t["r2"][1])  # the write waits for them
    assert t["r3"][0] >= t["w"][1]  # and the read after it waits for the write


async def test_two_writes_run_in_order_not_together(slow_harness):
    async with slow_harness() as intervals:
        await run(calls(("write", "w1"), ("write", "w2")))
        t = intervals()
    assert t["w1"][1] <= t["w2"][0]


async def test_results_keep_call_order_even_when_a_later_call_finishes_first(slow_harness):
    async with slow_harness():
        trace, history = await run(calls(("read", "slowpoke", 0.4), ("read", "quick", 0.0)))
    assert [t["result"] for t in trace] == ["read:slowpoke", "read:quick"]
    tool_messages = [m for m in history if isinstance(m, dict) and m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_messages] == [
        ("c0", "read:slowpoke"),
        ("c1", "read:quick"),
    ]


async def test_one_crashing_call_does_not_sink_its_siblings(slow_harness, monkeypatch):
    real = harness.call_tool

    async def flaky(name, args):
        if args["label"] == "bad":
            raise ConnectionError("pipe closed")
        return await real(name, args)

    monkeypatch.setattr(harness, "call_tool", flaky)
    async with slow_harness():
        trace, _ = await run(calls(("read", "ok1"), ("read", "bad"), ("read", "ok2")))
    assert [t["result"] for t in trace] == [
        "read:ok1",
        "error: tool call failed: ConnectionError: pipe closed",
        "read:ok2",
    ]


async def test_approvals_are_asked_one_at_a_time_in_order_and_denials_skip_the_call(
    slow_harness, monkeypatch
):
    monkeypatch.setattr(harness, "approval_mode", "all")
    asked = []

    async def approver(name, args):
        asked.append(args["label"])
        return args["label"] != "b"  # deny b

    async with slow_harness() as intervals:
        trace, _ = await run(calls(("read", "a"), ("read", "b"), ("read", "c")), approve=approver)
        t = intervals()
    assert asked == ["a", "b", "c"]
    assert set(t) == {"a", "c"}  # b never reached the server
    assert [bool(x.get("denied")) for x in trace] == [False, True, False]
    assert overlap(t["a"], t["c"])  # the approved reads still ran together
