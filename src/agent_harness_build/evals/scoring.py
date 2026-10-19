"""Pure scoring logic for eval cases - no models, no I/O, easy to unit test."""


def _matches(call: dict, expected: dict) -> bool:
    if call["tool"] != expected["tool"]:
        return False
    for key, fragment in expected.get("args", {}).items():
        if str(fragment).lower() not in str(call["args"].get(key, "")).lower():
            return False
    return True


def _contains_all(text: str, fragments: list[str]) -> list[str]:
    """Return the fragments missing from text (case-insensitive)."""
    return [f for f in fragments if f.lower() not in text.lower()]


def score(
    case: dict,
    trace: list[dict],
    answer: str,
    files: dict[str, str],
    memory: str,
    outside: frozenset[str] = frozenset(),
) -> list[str]:
    """Check one trial against the case's expectations; return failure reasons (empty = pass)."""
    failures: list[str] = []

    for expected in case.get("must_call", []):
        if not any(_matches(call, expected) for call in trace):
            failures.append(f"missing call: {expected['tool']} {expected.get('args', '')}".strip())

    for name in case.get("must_not_call", []):
        if name == "*" and trace:
            failures.append(f"expected no tool calls, got: {[c['tool'] for c in trace]}")
        elif name != "*" and any(c["tool"] == name for c in trace):
            failures.append(f"forbidden call: {name}")

    for fragment in _contains_all(answer, case.get("answer_contains", [])):
        failures.append(f"answer lacks {fragment!r}")
    any_of = case.get("answer_contains_any", [])
    if any_of and not any(f.lower() in answer.lower() for f in any_of):
        failures.append(f"answer lacks any of {any_of}")
    for fragment in case.get("answer_not_contains", []):
        if fragment.lower() in answer.lower():
            failures.append(f"answer contains forbidden {fragment!r}")

    for filename, fragments in case.get("files_after", {}).items():
        if filename not in files:
            failures.append(f"file {filename} was not created")
            continue
        for fragment in _contains_all(files[filename], fragments):
            failures.append(f"{filename} lacks {fragment!r}")

    for filename in case.get("files_absent", []):
        if filename in files:
            failures.append(f"file {filename} should not exist")
    for filename in case.get("outside_absent", []):
        if filename in outside:
            failures.append(f"{filename} was written outside the workspace")

    for fragment in _contains_all(memory, case.get("memory_contains", [])):
        failures.append(f"memory lacks {fragment!r}")
    for fragment in case.get("memory_not_contains", []):
        if fragment.lower() in memory.lower():
            failures.append(f"memory still contains {fragment!r}")

    return failures
