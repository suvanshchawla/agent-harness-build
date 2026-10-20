"""Approval prompts: risky tool calls need the user's OK before they run.

Everything runs against a real workspace server and a scripted fake model.
"""

import builtins

import pytest
from fakes import FakeClient, reply, tool_call
from mcp.types import ToolAnnotations

from agent_harness_build import harness

WRITE = {"filename": "todo.md", "content": "milk"}


@pytest.fixture(autouse=True)
def reset_approval_state(monkeypatch):
    monkeypatch.setattr(harness, "approval_mode", "writes")
    monkeypatch.setattr(harness, "always_allowed", set())


class Recorder:
    """An approver that records what it was asked and answers from a script."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.asked.append((name, args))
        return self.answers.pop(0) if self.answers else True


async def run(connected, script, approver, history=None):
    trace: list = []
    fake = FakeClient(*script)
    answer = await harness.run_agent(
        "go",
        history=history if history is not None else [],
        trace=trace,
        client=fake,
        approve=approver,
        verbose=False,
    )
    return answer, trace, fake


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


async def test_server_marks_tools_with_the_right_risk_hints(connected_harness):
    async with connected_harness():
        ann = harness.tool_annotations
        assert ann["workspace__read_file"].read_only_hint is True
        assert ann["workspace__list_files"].read_only_hint is True
        assert ann["workspace__write_file"].destructive_hint is True
        assert ann["workspace__forget_memory"].destructive_hint is True
        assert ann["workspace__save_memory"].destructive_hint is False


async def test_writes_mode_asks_only_for_destructive_tools(connected_harness):
    async with connected_harness():
        assert harness.needs_approval("workspace__write_file")
        assert harness.needs_approval("workspace__forget_memory")
        assert not harness.needs_approval("workspace__read_file")
        assert not harness.needs_approval("workspace__list_files")
        assert not harness.needs_approval("workspace__save_memory")  # only ever adds


def test_unannotated_tools_are_treated_as_risky(monkeypatch):
    monkeypatch.setitem(harness.tool_annotations, "other__mystery", None)
    monkeypatch.setitem(harness.tool_annotations, "other__partial", ToolAnnotations(read_only_hint=False))
    assert harness.needs_approval("other__mystery")
    assert harness.needs_approval("other__partial")  # destructive_hint unset means assume destructive


def test_all_and_none_modes(monkeypatch):
    monkeypatch.setitem(harness.tool_annotations, "s__read", ToolAnnotations(read_only_hint=True))
    monkeypatch.setattr(harness, "approval_mode", "all")
    assert harness.needs_approval("s__read")
    monkeypatch.setattr(harness, "approval_mode", "none")
    assert not harness.needs_approval("s__anything")


def test_always_allowed_skips_the_prompt(monkeypatch):
    monkeypatch.setitem(harness.tool_annotations, "s__write", None)
    assert harness.needs_approval("s__write")
    harness.always_allowed.add("s__write")
    assert not harness.needs_approval("s__write")


# ---------------------------------------------------------------------------
# In the agent loop
# ---------------------------------------------------------------------------


async def test_denied_call_does_not_run_and_the_model_is_told(connected_harness):
    script = [
        reply(tool_calls=[tool_call("workspace__write_file", WRITE)]),
        reply("Okay, I didn't write it."),
    ]
    approver = Recorder(False)
    async with connected_harness() as (workspace, _):
        answer, trace, fake = await run(connected_harness, script, approver)

        assert not (workspace / "todo.md").exists()  # the server never saw the call
    assert approver.asked == [("workspace__write_file", WRITE)]
    assert trace == [
        {"tool": "workspace__write_file", "args": WRITE, "result": harness.DENIED, "denied": True}
    ]
    told = [m for m in fake.requests[1] if isinstance(m, dict) and m["role"] == "tool"]
    assert told[0]["content"] == harness.DENIED
    assert answer == "Okay, I didn't write it."


async def test_approved_call_runs_normally(connected_harness):
    script = [reply(tool_calls=[tool_call("workspace__write_file", WRITE)]), reply("done")]
    approver = Recorder(True)
    async with connected_harness() as (workspace, _):
        _, trace, _ = await run(connected_harness, script, approver)
        assert (workspace / "todo.md").read_text() == "milk"
    assert len(approver.asked) == 1
    assert "denied" not in trace[0] and trace[0]["result"] == "wrote todo.md"


async def test_read_only_tools_never_prompt(connected_harness):
    script = [
        reply(
            tool_calls=[
                tool_call("workspace__list_files", {}),
                tool_call("workspace__read_file", {"filename": "x"}, "c2"),
            ]
        ),
        reply("ok"),
    ]
    approver = Recorder()
    async with connected_harness():
        await run(connected_harness, script, approver)
    assert approver.asked == []


async def test_a_denial_does_not_block_the_other_calls_in_the_same_message(connected_harness):
    script = [
        reply(
            tool_calls=[
                tool_call("workspace__write_file", {"filename": "a.txt", "content": "1"}, "c1"),
                tool_call("workspace__write_file", {"filename": "b.txt", "content": "2"}, "c2"),
            ]
        ),
        reply("done"),
    ]
    approver = Recorder(False, True)  # deny the first, allow the second
    async with connected_harness() as (workspace, _):
        _, trace, _ = await run(connected_harness, script, approver)
        assert sorted(p.name for p in workspace.iterdir()) == ["b.txt"]
    assert [bool(t.get("denied")) for t in trace] == [True, False]


async def test_no_approver_means_everything_is_approved(connected_harness):
    script = [reply(tool_calls=[tool_call("workspace__write_file", WRITE)]), reply("done")]
    async with connected_harness() as (workspace, _):
        await run(connected_harness, script, None)
        assert (workspace / "todo.md").exists()


async def test_none_mode_skips_prompts_in_the_loop(connected_harness, monkeypatch):
    monkeypatch.setattr(harness, "approval_mode", "none")
    script = [reply(tool_calls=[tool_call("workspace__write_file", WRITE)]), reply("done")]
    approver = Recorder(False)
    async with connected_harness() as (workspace, _):
        await run(connected_harness, script, approver)
        assert (workspace / "todo.md").exists()
    assert approver.asked == []


# ---------------------------------------------------------------------------
# The interactive prompt and the /approve command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("y", True), ("YES", True), (" yes ", True), ("n", False), ("", False), ("maybe", False)],
)
async def test_ask_user_parses_answers(monkeypatch, capsys, typed, expected):
    monkeypatch.setattr(builtins, "input", lambda _: typed)
    assert await harness.ask_user("workspace__write_file", WRITE) is expected
    out = capsys.readouterr().out
    assert "[approval needed] workspace__write_file" in out and "filename: todo.md" in out


async def test_ask_user_always_remembers_the_tool(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _: "a")
    assert await harness.ask_user("workspace__write_file", WRITE) is True
    assert "workspace__write_file" in harness.always_allowed


async def test_approve_command_switches_modes(capsys):
    assert await harness.handle_command("/approve all")
    assert harness.approval_mode == "all"
    await harness.handle_command("/approve bogus")
    assert harness.approval_mode == "all"  # unchanged
    assert "choose from" in capsys.readouterr().out
    await harness.handle_command("/approve writes")
    assert harness.approval_mode == "writes"
