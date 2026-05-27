import json
import os
import sys

_src_root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "")
if _src_root and _src_root not in sys.path:
    sys.path.insert(0, _src_root)
sys.path = [p for p in sys.path if p not in ("", ".")]

from tui_gateway import server
from tui_gateway.server import dispatch, write_json
from tui_gateway.transport import TeeTransport

_ALLOWED_METHODS = frozenset(
    {
        "agents.list",
        "browser.manage",
        "cli.exec",
        "clarify.respond",
        "config.get",
        "config.set",
        "cron.manage",
        "delegation.pause",
        "delegation.status",
        "image.attach",
        "model.disconnect",
        "model.options",
        "model.save_key",
        "process.stop",
        "prompt.submit",
        "reload.env",
        "reload.mcp",
        "rollback.diff",
        "rollback.list",
        "rollback.restore",
        "secret.respond",
        "session.close",
        "session.compress",
        "session.interrupt",
        "session.resume",
        "session.steer",
        "shell.exec",
        "skills.manage",
        "skills.reload",
        "subagent.interrupt",
        "sudo.respond",
        "terminal.resize",
        "tools.configure",
        "tools.list",
        "tools.show",
        "toolsets.list",
        "voice.record",
        "voice.toggle",
        "voice.tts",
    }
)


def _install_sidecar_publisher() -> None:
    url = os.environ.get("HERMES_TUI_SIDECAR_URL")
    if not url:
        return

    from tui_gateway.event_publisher import WsPublisherTransport

    server._stdio_transport = TeeTransport(
        server._stdio_transport, WsPublisherTransport(url)
    )


def _discover_mcp() -> None:
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        has_mcp_servers = isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        has_mcp_servers = True
    if not has_mcp_servers:
        return
    try:
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()
    except Exception:
        pass


def main() -> None:
    _install_sidecar_publisher()
    _discover_mcp()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            if not write_json(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "parse error"},
                    "id": None,
                }
            ):
                sys.exit(0)
            continue

        method = req.get("method", "")
        if method not in _ALLOWED_METHODS:
            if not write_json(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"internal worker method not allowed: {method}",
                    },
                    "id": req.get("id"),
                }
            ):
                sys.exit(0)
            continue

        resp = dispatch(req)
        if resp is not None and not write_json(resp):
            sys.exit(0)


if __name__ == "__main__":
    main()
