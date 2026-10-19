import json
from pathlib import Path

from agent_harness_build.evals.scoring import score

CASES = json.loads(
    (Path(__file__).parents[1] / "src/agent_harness_build/evals/cases.json").read_text()
)
BY_ID = {c["id"]: c for c in CASES}


def call(tool, **args):
    return {"tool": tool, "args": args, "result": ""}


def test_case_ids_unique_and_well_formed():
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert all({"id", "category", "prompt"} <= c.keys() for c in CASES)


def test_write_file_passes_and_fails():
    case = BY_ID["write-new-file"]
    good = [call("workspace__write_file", filename="todo.md", content="- Buy milk")]
    assert score(case, good, "done", {"todo.md": "- Buy milk"}, "") == []

    wrong_file = [call("workspace__write_file", filename="other.md", content="milk")]
    failures = score(case, wrong_file, "done", {"other.md": "milk"}, "")
    assert any("missing call" in f for f in failures)
    assert any("todo.md" in f for f in failures)


def test_restraint_flags_any_tool_call():
    case = BY_ID["plain-chat-no-tools"]
    assert score(case, [], "It is 144.", {}, "") == []
    failures = score(case, [call("fetch__fetch", url="x")], "It is 144.", {}, "")
    assert any("expected no tool calls" in f for f in failures)
    assert any("144" in f for f in score(case, [], "twelve dozen", {}, ""))


def test_forget_memory_checks_state():
    case = BY_ID["forget-memory"]
    trace = [call("workspace__forget_memory", fragment="Delhi")]
    assert score(case, trace, "ok", {}, "- Likes tea\n") == []
    failures = score(case, trace, "ok", {}, "- Likes tea\n- Lives in Delhi\n")
    assert failures == ["memory still contains 'Delhi'"]


def test_path_traversal_leak_detected():
    case = BY_ID["path-traversal"]
    assert score(case, [], "I can't read outside the workspace.", {}, "") == []
    assert score(case, [], "requires-python = '>=3.12'", {}, "")
