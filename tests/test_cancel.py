"""Cancelling a turn (Ctrl+C): running tools finish, unstarted ones are skipped, history stays valid.

Cancellation is driven by cancelling the run_agent task directly, the same thing the SIGINT
handler in chat() does. The slow test server's call log says when a tool call has started.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fakes import FakeClient, reply, tool_call

from agent_harness_build import harness


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(harness, "approval_mode", "writes")
    monkeypatch.setattr(harness, "always_allowed", set())


async def wait_for_starts(log, count=1):
    async def poll():
        while not log.exists() or log.read_text().count("start") < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), 10)


def started_labels(log):
    return [line.split()[2] for line in log.read_text().splitlines() if line.startswith("start")]


def tool_messages(history):
    return [m for m in history if isinstance(m, dict) and m["role"] == "tool"]


def slow_calls(*specs):
    return [
        tool_call(f"slow__{tool}", {"label": label, "delay": delay}, f"c{i}")
        for i, (tool, label, delay) in enumerate(specs)
    ]


class HangingClient:
    """A model that answers its scripted replies, then hangs forever on the next call."""

    def __init__(self, *script):
        self.fake = FakeClient(*script) if script else None
        self.remaining = len(script)
        self.hanging = asyncio.Event()  # set once a call is stuck waiting
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        if self.remaining == 0:
            self.hanging.set()
            await asyncio.Event().wait()
        self.remaining -= 1
        return await self.fake._create(**kwargs)


async def test_cancel_during_the_model_call_undoes_the_turn(connected_harness):
    client = HangingClient()
    history: list = [{"role": "system", "content": "sys"}]
    async with connected_harness():
        turn = asyncio.ensure_future(harness.run_agent("hi", history=history, client=client, verbose=False))
        await asyncio.wait_for(client.hanging.wait(), 10)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
    assert history == [{"role": "system", "content": "sys"}]  # nothing ran, so it never happened


async def test_a_running_tool_finishes_and_the_model_is_told_about_it(slow_harness):
    fake = FakeClient(reply(tool_calls=slow_calls(("write", "a", 0.5))), reply("unused"))
    history: list = []
    trace: list = []
    async with slow_harness() as intervals:
        turn = asyncio.ensure_future(
            harness.run_agent("write", history=history, trace=trace, client=fake, verbose=False, approve=None)
        )
        await wait_for_starts(intervals.log)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert "a" in intervals()  # the call ran to its end, it was not abandoned halfway
    assert [m["content"] for m in tool_messages(history)] == ["write:a"]
    assert trace[0]["result"] == "write:a"
    assert history[-1] == {"role": "assistant", "content": harness.CANCEL_NOTE}
    assert len(fake.requests) == 1  # the turn stopped; the model was not asked again


async def test_calls_that_had_not_started_are_skipped_and_reported(slow_harness):
    fake = FakeClient(reply(tool_calls=slow_calls(("write", "a", 0.5), ("write", "b", 0.1))))
    history: list = []
    async with slow_harness() as intervals:
        turn = asyncio.ensure_future(harness.run_agent("go", history=history, client=fake, verbose=False))
        await wait_for_starts(intervals.log)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert started_labels(intervals.log) == ["a"]  # b never started
    results = [m["content"] for m in tool_messages(history)]
    assert results == ["write:a", harness.SKIPPED]  # both calls have a result, so the history is valid
    assert [m["tool_call_id"] for m in tool_messages(history)] == ["c0", "c1"]


async def test_a_second_cancel_stops_waiting_for_a_slow_tool(slow_harness):
    fake = FakeClient(reply(tool_calls=slow_calls(("write", "slow", 30))))
    history: list = []
    async with slow_harness() as intervals:
        turn = asyncio.ensure_future(harness.run_agent("go", history=history, client=fake, verbose=False))
        await wait_for_starts(intervals.log)
        turn.cancel()
        await asyncio.sleep(0.2)  # still waiting for the tool, as the first cancel promises
        assert not turn.done()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 5)
    assert [m["content"] for m in tool_messages(history)] == [harness.SKIPPED]
    assert history[-1] == {"role": "assistant", "content": harness.CANCEL_NOTE}


async def test_cancel_after_a_tool_ran_keeps_that_work_in_the_history(slow_harness):
    client = HangingClient(reply(tool_calls=slow_calls(("write", "a", 0.05))))
    history: list = []
    async with slow_harness() as intervals:
        turn = asyncio.ensure_future(harness.run_agent("go", history=history, client=client, verbose=False))
        await asyncio.wait_for(client.hanging.wait(), 10)  # the tool is done; the model call is stuck
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert "a" in intervals()
    assert [m["content"] for m in tool_messages(history)] == ["write:a"]  # not forgotten
    assert history[-1] == {"role": "assistant", "content": harness.CANCEL_NOTE}


async def test_cancelling_while_an_approval_is_open_means_the_call_never_runs(slow_harness):
    asked, answer = asyncio.Event(), asyncio.Event()

    async def approve(name, args):
        asked.set()
        await answer.wait()  # the user is still looking at the prompt
        return True

    fake = FakeClient(reply(tool_calls=slow_calls(("write", "a", 0.1))))
    history: list = []
    async with slow_harness() as intervals:
        turn = asyncio.ensure_future(
            harness.run_agent("go", history=history, client=fake, verbose=False, approve=approve)
        )
        await asyncio.wait_for(asked.wait(), 10)
        turn.cancel()
        await asyncio.sleep(0.1)
        answer.set()  # they answer "yes" after pressing Ctrl+C
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, 5)
        assert not intervals.log.exists()  # the approved call was never run
    assert [m["content"] for m in tool_messages(history)] == [harness.SKIPPED]


async def test_execute_tool_calls_with_stop_already_set_runs_nothing(slow_harness):
    async with slow_harness() as intervals:
        stop = asyncio.Event()
        stop.set()
        out = await harness.execute_tool_calls(
            [("slow__write", {"label": "a"}), ("slow__read", {"label": "b"})], stop=stop, verbose=False
        )
        assert out == [(harness.SKIPPED, False)] * 2
        assert not intervals.log.exists()
