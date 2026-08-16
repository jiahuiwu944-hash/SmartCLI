from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import paicli.mcp.client as mcp_client
from paicli.config import load_config
from paicli.mcp import McpClientManager
from paicli.mcp.config import write_chrome_devtools_config
from paicli.mcp.server import _handle_request
from paicli.tools.base import ToolContext


def test_chrome_devtools_config_uses_isolated_profile(tmp_path):
    path = write_chrome_devtools_config(scope_root=tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))

    args = config["mcpServers"]["chrome-devtools"]["args"]
    assert "--isolated" in args


def test_mcp_tools_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    async def run():
        return await _handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            str(tmp_path),
        )

    response = asyncio.run(run())
    tools = response["result"]["tools"]
    assert any(tool["name"] == "read_file" for tool in tools)
    assert any(tool["name"] == "execute_command" for tool in tools)


def test_mcp_client_registers_and_calls_stdio_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")

@mcp.tool()
def echo(text: str) -> str:
    return "echo:" + text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".paicli").mkdir()
    (tmp_path / ".paicli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fake": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        names = [tool.name for tool in tools]
        tool = next(item for item in tools if item.name == "mcp__fake__echo")
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        result = await tool.execute({"text": "ok"}, ToolContext(cwd=str(tmp_path), config=config))
        await manager.close()
        return names, result

    names, result = asyncio.run(run())
    assert "mcp__fake__echo" in names
    assert result.content == "echo:ok"


def test_mcp_client_suppresses_stdio_server_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    server = tmp_path / "noisy_mcp_server.py"
    server.write_text(
        """
import sys
from mcp.server.fastmcp import FastMCP

sys.stderr.write("NOISY_MCP_STARTUP\\n")
sys.stderr.flush()

mcp = FastMCP("noisy")

@mcp.tool()
def echo(text: str) -> str:
    return text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".paicli").mkdir()
    (tmp_path / ".paicli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "noisy": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        await manager.close()
        return tools

    tools = asyncio.run(run())

    assert any(tool.name == "mcp__noisy__echo" for tool in tools)
    captured = capsys.readouterr()
    assert "NOISY_MCP_STARTUP" not in captured.err


def test_mcp_client_reuses_one_persistent_stdio_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    starts = tmp_path / "starts.txt"
    server = tmp_path / "stateful_mcp_server.py"
    server.write_text(
        """
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

Path(os.environ["MCP_START_FILE"]).open("a", encoding="utf-8").write("started\\n")
mcp = FastMCP("stateful")
calls = 0

@mcp.tool()
def increment() -> int:
    global calls
    calls += 1
    return calls

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".paicli").mkdir()
    (tmp_path / ".paicli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "stateful": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                        "env": {"MCP_START_FILE": str(starts)},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        tool = next(item for item in tools if item.name == "mcp__stateful__increment")
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        context = ToolContext(cwd=str(tmp_path), config=config)
        first = await tool.execute({}, context)
        second = await tool.execute({}, context)
        connected = manager.connected_servers
        await manager.close()
        return first, second, connected

    first, second, connected = asyncio.run(run())

    assert first.content == "1"
    assert second.content == "2"
    assert connected == ("stateful",)
    assert starts.read_text(encoding="utf-8").splitlines() == ["started"]


def test_mcp_client_reconnects_once_after_transport_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    failure_marker = tmp_path / "failed_once.txt"
    starts = tmp_path / "starts.txt"
    server = tmp_path / "flaky_mcp_server.py"
    server.write_text(
        """
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

Path(os.environ["MCP_START_FILE"]).open("a", encoding="utf-8").write("started\\n")
mcp = FastMCP("flaky")

@mcp.tool()
def flaky() -> str:
    marker = Path(os.environ["MCP_FAILURE_MARKER"])
    if not marker.exists():
        marker.write_text("failed", encoding="utf-8")
        os._exit(17)
    return "recovered"

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / ".paicli").mkdir()
    (tmp_path / ".paicli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "flaky": {
                        "type": "stdio",
                        "command": "python",
                        "args": [str(server)],
                        "env": {
                            "MCP_FAILURE_MARKER": str(failure_marker),
                            "MCP_START_FILE": str(starts),
                        },
                        "max_retries": 1,
                        "retry_base_delay": 0.01,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        tool = next(item for item in tools if item.name == "mcp__flaky__flaky")
        config = load_config(project_root=tmp_path)
        config.policy.hitl_mode = "never"
        result = await tool.execute({}, ToolContext(cwd=str(tmp_path), config=config))
        await manager.close()
        return result

    result = asyncio.run(run())

    assert result.content == "recovered"
    assert starts.read_text(encoding="utf-8").splitlines() == ["started", "started"]


def test_mcp_servers_initialize_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".paicli").mkdir()
    (tmp_path / ".paicli" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "first": {"command": "unused"},
                    "second": {"command": "unused"},
                }
            }
        ),
        encoding="utf-8",
    )
    active = 0
    peak = 0
    both_started = asyncio.Event()

    class FakeConnection:
        def __init__(self, spec, project_root):
            self.spec = spec
            self.project_root = project_root
            self.capabilities = SimpleNamespace(tools=None, resources=None, prompts=None)

        async def start(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            active -= 1

        async def close(self):
            return None

    monkeypatch.setattr(mcp_client, "_PersistentMcpConnection", FakeConnection)

    async def run():
        manager = McpClientManager(tmp_path)
        tools = await manager.load_tools()
        connected = set(manager.connected_servers)
        await manager.close()
        return tools, connected

    tools, connected = asyncio.run(run())

    assert tools == []
    assert connected == {"first", "second"}
    assert peak == 2
