"""A scripted stand-in for AsyncOpenAI, shared by the agent-loop and runner tests."""

import json
from types import SimpleNamespace


def tool_call(name, args, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content=None, tool_calls=None, reasoning=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=reasoning)


def chunk(content=None, tool_calls=None, reasoning=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def fragment(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    """Stands in for AsyncOpenAI. Returns scripted messages, then repeats the last one."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[list] = []  # a snapshot of the conversation sent on each call
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model, messages, tools, stream=False):
        self.requests.append(list(messages))
        message = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(message, Exception):  # a scripted API failure, e.g. a 400
            raise message
        if stream:
            return self._stream(message)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def _stream(self, message, size=3):
        """Replay a scripted message as chunks: text in pieces of `size` characters, and each
        tool call split so the name comes first and its JSON arguments trickle in after."""
        thinking = getattr(message, "reasoning", None) or ""
        for i in range(0, len(thinking), size):
            yield chunk(reasoning=thinking[i : i + size])
        text = message.content or ""
        for i in range(0, len(text), size):
            yield chunk(content=text[i : i + size])
        for index, call in enumerate(message.tool_calls or []):
            yield chunk(tool_calls=[fragment(index, call_id=call.id, name=call.function.name)])
            args = call.function.arguments
            for i in range(0, len(args), size):
                yield chunk(tool_calls=[fragment(index, arguments=args[i : i + size])])
        if getattr(message, "fail", None):  # the connection drops mid-reply
            raise message.fail
        yield SimpleNamespace(choices=[])  # a usage-only chunk, as some providers send
