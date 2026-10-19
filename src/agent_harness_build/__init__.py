import asyncio


def main() -> None:
    from agent_harness_build.harness import main as run

    asyncio.run(run())
