"""Run the eval cases against one or more models and print a comparison table.

    uv run harness-evals                       # every model that has credentials
    uv run harness-evals --models local        # just one
    uv run harness-evals --trials 3            # repeat each case to smooth out noise
    uv run harness-evals --case write-new-file

Each trial runs the real harness agent loop against real MCP servers, but with a
throwaway workspace and memory file, so your own data is never touched.
"""

import argparse
import asyncio
import json
import shutil
import socket
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from agent_harness_build import harness
from agent_harness_build.evals.scoring import score

CASES_FILE = Path(__file__).parent / "cases.json"
RESULTS_DIR = Path(__file__).parent / "results"


def online() -> bool:
    try:
        socket.create_connection(("example.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


def usable_models(requested: list[str] | None) -> list[str]:
    names = requested or list(harness.MODELS)
    usable = []
    for name in names:
        if name not in harness.MODELS:
            sys.exit(f"unknown model '{name}' - choose from {list(harness.MODELS)}")
        cfg = harness.MODELS[name]
        if not cfg.api_key:
            print(f"  skipping '{name}': no API key set")
        else:
            usable.append(name)
    return usable


def write_servers_config(tmp: Path, workspace: Path, memory: Path) -> Path:
    """Same servers as mcp_servers.json, but the workspace server points at the temp dirs."""
    config = json.loads(harness.MCP_CONFIG.read_text(encoding="utf-8"))
    config["mcpServers"]["workspace"]["env"] = {
        "AGENT_WORKSPACE": str(workspace),
        "AGENT_MEMORY_FILE": str(memory),
    }
    path = tmp / "servers.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def reset_state(case: dict, workspace: Path, memory: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    for name, content in case.get("files", {}).items():
        (workspace / name).write_text(content, encoding="utf-8")
    memory.unlink(missing_ok=True)
    if "memory" in case:
        memory.write_text(case["memory"], encoding="utf-8")


def snapshot(workspace: Path, memory: Path) -> tuple[dict[str, str], str, frozenset[str]]:
    files = {p.name: p.read_text(encoding="utf-8") for p in workspace.iterdir() if p.is_file()}
    outside = frozenset(p.name for p in workspace.parent.iterdir())  # anything that escaped
    return files, memory.read_text(encoding="utf-8") if memory.exists() else "", outside


async def run_trial(case: dict, model: str, workspace: Path, memory: Path) -> dict:
    reset_state(case, workspace, memory)
    history = [
        {"role": "system", "content": harness.SYSTEM_PROMPT.format(memory=await harness.read_memory())}
    ]
    trace: list[dict] = []
    usage: dict = {}
    start = time.perf_counter()
    try:
        answer = await harness.run_agent(
            case["prompt"], history=history, model=model, trace=trace, verbose=False, usage=usage
        )
    except Exception as e:  # network down, bad key, model rejects tools...
        return {
            "status": "error",
            "failures": [f"{type(e).__name__}: {e}"],
            "trace": trace,
            "answer": "",
            "seconds": time.perf_counter() - start,
            **usage,
        }
    seconds = time.perf_counter() - start

    files, mem, outside = snapshot(workspace, memory)
    failures = score(case, trace, answer, files, mem, outside)
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "trace": trace,
        "answer": answer,
        "seconds": seconds,
        **usage,
    }


def print_table(results: list[dict], models: list[str]) -> None:
    categories = sorted({r["category"] for r in results})
    tiers = sorted({r["tier"] for r in results})
    columns = [*categories, *tiers]
    stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model -> column -> [pass, total]
    errors: dict = defaultdict(int)
    seconds: dict = defaultdict(list)
    for r in results:
        for column in (r["category"], r["tier"]):
            cell = stats[r["model"]][column]
            cell[1] += 1
            cell[0] += r["status"] == "pass"
        errors[r["model"]] += r["status"] == "error"
        seconds[r["model"]].append(r["seconds"])

    header = ["model", *columns, "overall", "errors", "avg s"]
    rows = []
    for m in models:
        cells = [f"{stats[m][c][0]}/{stats[m][c][1]}" if stats[m][c][1] else "-" for c in columns]
        passed = sum(stats[m][c][0] for c in tiers)
        total = sum(stats[m][c][1] for c in tiers)
        rows.append(
            [
                m,
                *cells,
                f"{passed}/{total}",
                str(errors[m]),
                f"{sum(seconds[m]) / len(seconds[m]):.1f}" if seconds[m] else "-",
            ]
        )
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def print_cost(results: list[dict], models: list[str]) -> None:
    """Speed and token use per model, averaged per trial.

    Token columns average only over trials where the provider reported a count, and show "-" if
    none did (e.g. an endpoint that sends no usage), rather than a made-up zero.
    """

    def mean(values: list[float], fmt: str) -> str:
        return format(sum(values) / len(values), fmt) if values else "-"

    header = ["model", "trials", "avg s", "median s", "max s", "model calls", "prompt tok", "output tok"]
    rows = []
    for m in models:
        trials = [r for r in results if r["model"] == m]
        if not trials:
            continue
        secs = sorted(r["seconds"] for r in trials)
        middle = len(secs) // 2
        median = secs[middle] if len(secs) % 2 else (secs[middle - 1] + secs[middle]) / 2
        rows.append(
            [
                m,
                str(len(trials)),
                f"{sum(secs) / len(secs):.1f}",
                f"{median:.1f}",
                f"{secs[-1]:.1f}",
                mean([r["model_calls"] for r in trials if "model_calls" in r], ".1f"),
                mean([r["prompt_tokens"] for r in trials if "prompt_tokens" in r], ".0f"),
                mean([r["completion_tokens"] for r in trials if "completion_tokens" in r], ".0f"),
            ]
        )
    if not rows:
        return
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    print()
    for row in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))


def first_reason(trials: list[dict]) -> str:
    """Why the first non-passing trial failed, shortened for one line."""
    for r in trials:
        if r["status"] != "pass":
            return "; ".join(r.get("failures") or [r["status"]])[:110]
    return ""


def print_reliability(results: list[dict], models: list[str]) -> None:
    """With repeated trials, say which cases are stable and which are flaky.

    The table pools every trial, so a case that passes 2 times in 3 looks the same as three
    different cases each passing once. A flaky case is one with both passes and non-passes
    (a crash counts as a non-pass).
    """
    by_case: dict = defaultdict(lambda: defaultdict(list))  # model -> case -> trials
    for r in results:
        by_case[r["model"]][r["case"]].append(r)
    if not any(len(trials) > 1 for cases in by_case.values() for trials in cases.values()):
        return

    for model in models:
        cases = by_case[model]
        if not cases:
            continue
        rate = {c: (sum(r["status"] == "pass" for r in t), len(t)) for c, t in cases.items()}
        flaky = sorted((c for c, (p, n) in rate.items() if 0 < p < n), key=lambda c: rate[c][0] / rate[c][1])
        failing = sorted(c for c, (p, _) in rate.items() if p == 0)
        stable = len(rate) - len(flaky) - len(failing)
        print(
            f"\n{model}: {stable} always pass, {len(flaky)} flaky, {len(failing)} always fail"
            f" (of {len(rate)} cases)"
        )
        for label, names in (("flaky", flaky), ("fails", failing)):
            for case in names:
                passed, total = rate[case]
                print(f"  {label}  {case}  {passed}/{total}  {first_reason(cases[case])}")


async def main_async(args: argparse.Namespace) -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if args.category:
        cases = [c for c in cases if c["category"] in args.category]
    if args.tier:
        cases = [c for c in cases if c["tier"] in args.tier]
    if not cases:
        sys.exit("no cases selected")
    models = usable_models(args.models)
    if not models:
        sys.exit("no usable models - start Ollama or set API keys in .env")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        workspace, memory = tmp / "workspace", tmp / "memory.md"
        config = write_servers_config(tmp, workspace, memory)

        async with AsyncExitStack() as stack:
            await harness.connect_mcp(stack, config)
            connected = set(harness.sessions)
            has_network = online()
            results = []
            for model in models:
                print(f"\n== {model} ({harness.MODELS[model].model})")
                for case in cases:
                    if case.get("needs_network") and not has_network:
                        print(f"  skip  {case['id']} (no network)")
                        continue
                    missing = set(case.get("requires_servers", [])) - connected
                    if missing:
                        print(f"  skip  {case['id']} (server not connected: {sorted(missing)})")
                        continue
                    for trial in range(args.trials):
                        r = await run_trial(case, model, workspace, memory)
                        results.append(
                            {
                                "model": model,
                                "case": case["id"],
                                "category": case["category"],
                                "tier": case["tier"],
                                "trial": trial,
                                **r,
                            }
                        )
                        note = f"  - {'; '.join(r['failures'])}" if r["failures"] else ""
                        print(f"  {r['status']:<5} {case['id']} [{r['seconds']:.1f}s]{note}")

    print()
    print_table(results, models)
    print_cost(results, models)
    print_reliability(results, models)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull results: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on tool use through the harness.")
    parser.add_argument("--models", nargs="+", help="model names from the registry (default: all usable)")
    parser.add_argument("--trials", type=int, default=1, help="runs per case per model")
    parser.add_argument("--case", nargs="+", help="only run these case ids")
    parser.add_argument("--tier", nargs="+", choices=["easy", "hard"], help="only run these tiers")
    parser.add_argument("--category", nargs="+", help="only run these categories")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
