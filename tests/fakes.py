"""A scripted stand-in for AsyncOpenAI, shared by the agent-loop and runner tests."""

import json
from types import SimpleNamespace


def tool_call(name, args, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeClient:
    """Stands in for AsyncOpenAI. Returns scripted messages, then repeats the last one."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[list] = []  # a snapshot of the conversation sent on each call
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, tools):
        self.requests.append(list(messages))
        message = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
