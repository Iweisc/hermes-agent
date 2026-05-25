#!/usr/bin/env python3
import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
MARKERS = (
    "Hermes needs a model provider before the TUI can start a session.",
    "/help for commands",
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def run_step(argv: list[str], cwd: Path) -> None:
    result = subprocess.run(argv, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_step(["cargo", "build", "-q", "-p", "hermes-rs-cli", "--bin", "hermes"], repo_root)

    ui_dir = repo_root / "ui-tui"
    if not (ui_dir / "node_modules").is_dir():
        run_step(
            ["npm", "install", "--silent", "--no-fund", "--no-audit"],
            ui_dir,
        )
    run_step(["npm", "run", "build"], ui_dir)

    hermes_bin = repo_root / "target" / "debug" / "hermes"
    tmpdir = Path(tempfile.mkdtemp(prefix="hermes-rust-tui-smoke-"))
    hermes_home = tmpdir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "CI": "1",
            "HERMES_HOME": str(hermes_home),
            "HERMES_PYTHON": "/definitely/missing/python",
            "HERMES_SKIP_NODE_BOOTSTRAP": "1",
            "HERMES_TUI_INLINE": "1",
            "HERMES_TUI_STARTUP_TIMEOUT_MS": "60000",
            "TERM": "xterm-256color",
        }
    )

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        [str(hermes_bin), "--tui"],
        cwd=repo_root,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
    )
    os.close(slave_fd)

    chunks: list[str] = []
    deadline = time.time() + 60
    try:
        while time.time() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.5)
            if not readable:
                if proc.poll() is not None:
                    break
                continue
            data = os.read(master_fd, 65536)
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="ignore"))
            plain = strip_ansi("".join(chunks))
            flat = normalized(plain)
            if any(normalized(marker) in flat for marker in MARKERS):
                print("smoke_tui_rust_backend: ok")
                return 0
            if "gateway startup timed out" in plain.lower():
                print(plain)
                print("smoke_tui_rust_backend: backend startup timed out", file=sys.stderr)
                return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        os.close(master_fd)

    plain = strip_ansi("".join(chunks))
    if plain:
        print(plain)
    print("smoke_tui_rust_backend: did not observe a ready marker", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
