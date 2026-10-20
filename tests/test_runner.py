"""Tests for the eval runner: state handling, config, the results table, and full trials.

Trials run the real agent loop and a real workspace server against a scripted fake model, so
nothing here needs an API key or Ollama.
"""

import argparse
import json
from types import SimpleNamespace

import pytest
from fakes import FakeClient, reply, tool_call

from agent_harness_build import harness
from agent_harness_build.evals import runner

WRITE_TODO = {"filename": "todo.md", "content": "buy milk"}
WRITE_CASE = {
    "id": "write-todo",
    "category": "tool_use",
    "tier": "easy",
    "prompt": "add milk",
    "must_call": [{"tool": "workspace__write_file", "args": {"filename": "todo.md"}}],
    "files_after": {"todo.md": ["milk"]},
}


def use_fake_model(monkeypatch, *script):
    """Make every run_agent call build a fresh FakeClient with this script."""
    monkeypatch.setattr(harness, "AsyncOpenAI", lambda **_: FakeClient(*script))


# ---------------------------------------------------------------------------
# State handling
# ---------------------------------------------------------------------------


def test_reset_state_wipes_old_files_and_seeds_the_case(tmp_path):
    workspace, memory = tmp_path / "workspace", tmp_path / "memory.md"
    workspace.mkdir()
    (workspace / "leftover.md").write_text("from the last trial")
    memory.write_text("- old fact\n")

    runner.reset_state({"files": {"a.md": "A"}, "memory": "- Likes tea\n"}, workspace, memory)

    assert sorted(p.name for p in workspace.iterdir()) == ["a.md"]
    assert memory.read_text() == "- Likes tea\n"


def test_reset_state_without_memory_removes_the_memory_file(tmp_path):
    workspace, memory = tmp_path / "workspace", tmp_path / "memory.md"
    memory.write_text("- stale\n")
    runner.reset_state({}, workspace, memory)
    assert workspace.is_dir() and list(workspace.iterdir()) == []
    assert not memory.exists()


def test_snapshot_reports_files_memory_and_anything_outside(tmp_path):
    workspace, memory = tmp_path / "workspace", tmp_path / "memory.md"
    runner.reset_state({"files": {"a.md": "A"}, "memory": "- x\n"}, workspace, memory)
    (workspace / "subdir").mkdir()  # directories are not files
    (tmp_path / "escape.txt").write_text("leaked")

    files, mem, outside = runner.snapshot(workspace, memory)

    assert files == {"a.md": "A"}
    assert mem == "- x\n"
    assert "escape.txt" in outside


def test_snapshot_with_no_memory_file(tmp_path):
    workspace, memory = tmp_path / "workspace", tmp_path / "memory.md"
    runner.reset_state({}, workspace, memory)
    assert runner.snapshot(workspace, memory)[1] == ""


def test_write_servers_config_points_only_the_workspace_server_at_temp_dirs(tmp_path):
    workspace, memory = tmp_path / "w", tmp_path / "m.md"
    path = runner.write_servers_config(tmp_path, workspace, memory)

    servers = json.loads(path.read_text())["mcpServers"]
    assert servers["workspace"]["env"] == {
        "AGENT_WORKSPACE": str(workspace),
        "AGENT_MEMORY_FILE": str(memory),
    }
    original = json.loads(harness.MCP_CONFIG.read_text())["mcpServers"]
    assert "env" not in original["workspace"]  # the real config file is not modified
    assert {k: v for k, v in servers.items() if k != "workspace"} == {
        k: v for k, v in original.items() if k != "workspace"
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_usable_models_skips_models_without_a_key(monkeypatch, capsys):
    monkeypatch.setattr(
        harness,
        "MODELS",
        {
            "local": harness.ModelConfig(base_url="http://x", api_key="ollama", model="m"),
            "gpt": harness.ModelConfig(base_url="http://y", api_key="", model="g"),
        },
    )
    assert runner.usable_models(None) == ["local"]
    assert "skipping 'gpt'" in capsys.readouterr().out


def test_usable_models_rejects_unknown_names():
    with pytest.raises(SystemExit, match="unknown model 'nope'"):
        runner.usable_models(["nope"])


# ---------------------------------------------------------------------------
# The results table
# ---------------------------------------------------------------------------


def result(model, case, category, tier, status, seconds=1.0):
    return {
        "model": model,
        "case": case,
        "category": category,
        "tier": tier,
        "status": status,
        "seconds": seconds,
    }


def test_print_table_counts_by_category_tier_and_overall(capsys):
    results = [
        result("m", "a", "memory", "easy", "pass", 1.0),
        result("m", "b", "memory", "hard", "fail", 3.0),
        result("m", "c", "safety", "hard", "pass", 2.0),
        result("m", "d", "safety", "hard", "error", 2.0),
    ]
    runner.print_table(results, ["m"])
    header, _, row = capsys.readouterr().out.strip().splitlines()

    assert header.split() == ["model", "memory", "safety", "easy", "hard", "overall", "errors", "avg", "s"]
    assert row.split() == ["m", "1/2", "1/2", "1/1", "1/3", "2/4", "1", "2.0"]


def test_print_table_handles_a_model_whose_cases_were_all_skipped(capsys):
    runner.print_table([result("m1", "a", "memory", "easy", "pass")], ["m1", "m2"])
    row = capsys.readouterr().out.strip().splitlines()[-1]
    assert row.split() == ["m2", "-", "-", "0/0", "0", "-"]


# ---------------------------------------------------------------------------
# Trials: the real loop and server against a fake model
# ---------------------------------------------------------------------------


async def test_trial_passes_when_the_model_does_the_right_thing(connected_harness, monkeypatch):
    use_fake_model(
        monkeypatch, reply(tool_calls=[tool_call("workspace__write_file", WRITE_TODO)]), reply("done")
    )
    async with connected_harness() as (workspace, memory):
        r = await runner.run_trial(WRITE_CASE, "local", workspace, memory)

    assert r["status"] == "pass" and r["failures"] == []
    assert [t["tool"] for t in r["trace"]] == ["workspace__write_file"]
    assert r["answer"] == "done" and r["seconds"] >= 0


async def test_trial_fails_with_reasons_when_the_model_does_nothing(connected_harness, monkeypatch):
    use_fake_model(monkeypatch, reply("Sure, I did it."))
    async with connected_harness() as (workspace, memory):
        r = await runner.run_trial(WRITE_CASE, "local", workspace, memory)

    assert r["status"] == "fail"
    assert any("missing call" in f for f in r["failures"])
    assert any("todo.md was not created" in f or "todo.md" in f for f in r["failures"])


async def test_trial_marks_a_crashing_model_as_error_not_fail(connected_harness, monkeypatch):
    class Boom:
        def __init__(self, **_):
            async def create(**_):
                raise ConnectionError("ollama is not running")

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(harness, "AsyncOpenAI", Boom)
    async with connected_harness() as (workspace, memory):
        r = await runner.run_trial(WRITE_CASE, "local", workspace, memory)

    assert r["status"] == "error"
    assert r["failures"] == ["ConnectionError: ollama is not running"]
    assert r["trace"] == [] and r["answer"] == ""


async def test_each_trial_starts_from_the_cases_own_state(connected_harness, monkeypatch):
    first = {**WRITE_CASE, "id": "first"}
    second = {
        "id": "second",
        "category": "restraint",
        "tier": "easy",
        "prompt": "hi",
        "must_not_call": ["*"],
        "files": {"seed.md": "seed"},
        "memory": "- Likes tea\n",
    }
    async with connected_harness() as (workspace, memory):
        use_fake_model(
            monkeypatch, reply(tool_calls=[tool_call("workspace__write_file", WRITE_TODO)]), reply("ok")
        )
        await runner.run_trial(first, "local", workspace, memory)
        assert (workspace / "todo.md").exists()

        use_fake_model(monkeypatch, reply("hello"))
        r = await runner.run_trial(second, "local", workspace, memory)

        assert sorted(p.name for p in workspace.iterdir()) == ["seed.md"]  # todo.md is gone
        assert memory.read_text() == "- Likes tea\n"
        assert r["status"] == "pass"


async def test_the_model_sees_the_cases_starting_memory_in_its_system_prompt(connected_harness, monkeypatch):
    seen = []

    class Spy(FakeClient):
        async def _create(self, model, messages, tools):
            seen.append(messages[0]["content"])
            return await super()._create(model, messages, tools)

    monkeypatch.setattr(harness, "AsyncOpenAI", lambda **_: Spy(reply("green")))
    case = {
        "id": "c",
        "category": "memory",
        "tier": "easy",
        "prompt": "colour?",
        "memory": "- Favourite colour is green\n",
    }
    async with connected_harness() as (workspace, memory):
        await runner.run_trial(case, "local", workspace, memory)
    assert "Favourite colour is green" in seen[0]


# ---------------------------------------------------------------------------
# main_async, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """Isolate everything main_async touches: cases, server config, results dir, network."""
    cases = [
        WRITE_CASE,
        {**WRITE_CASE, "id": "needs-time", "requires_servers": ["time"]},
        {**WRITE_CASE, "id": "needs-net", "needs_network": True},
    ]
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(cases))
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "workspace": {
                        "command": "{python}",
                        "args": ["-m", "agent_harness_build.servers.workspace_server"],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(runner, "CASES_FILE", cases_file)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(harness, "MCP_CONFIG", config)
    monkeypatch.setattr(runner, "online", lambda: False)
    monkeypatch.setattr(
        harness, "MODELS", {"local": harness.ModelConfig(base_url="http://x", api_key="ollama", model="fake")}
    )
    use_fake_model(
        monkeypatch, reply(tool_calls=[tool_call("workspace__write_file", WRITE_TODO)]), reply("done")
    )
    for registry in (harness.TOOL_SCHEMAS, harness.tool_routes, harness.prompt_routes, harness.sessions):
        registry.clear()
    return tmp_path


def args(**overrides):
    base = {"models": None, "trials": 1, "case": None, "tier": None, "category": None}
    return argparse.Namespace(**{**base, **overrides})


async def test_main_runs_trials_skips_unavailable_cases_and_saves_results(runner_env, capsys):
    await runner.main_async(args(trials=2))
    out = capsys.readouterr().out

    assert "skip  needs-time (server not connected: ['time'])" in out
    assert "skip  needs-net (no network)" in out
    assert out.count("pass  write-todo") == 2

    saved = json.loads(next((runner_env / "results").glob("*.json")).read_text())
    assert [(r["case"], r["trial"], r["status"]) for r in saved] == [
        ("write-todo", 0, "pass"),
        ("write-todo", 1, "pass"),
    ]
    assert {"model", "category", "tier", "trace", "answer", "seconds", "failures"} <= saved[0].keys()


async def test_main_survives_a_model_with_every_case_skipped(runner_env, capsys):
    await runner.main_async(args(case=["needs-time", "needs-net"]))
    out = capsys.readouterr().out
    assert "full results:" in out
    assert json.loads(next((runner_env / "results").glob("*.json")).read_text()) == []


async def test_main_filters_and_rejects_empty_selection(runner_env):
    with pytest.raises(SystemExit, match="no cases selected"):
        await runner.main_async(args(case=["does-not-exist"]))
    with pytest.raises(SystemExit, match="no cases selected"):
        await runner.main_async(args(tier=["hard"]))
