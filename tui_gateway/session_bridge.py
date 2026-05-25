import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class _NullTransport:
    def write(self, obj: dict) -> bool:
        return True

    def close(self) -> None:
        return None


def _call(method: str, params: dict) -> dict:
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    )
    if response is None:
        raise RuntimeError(f"{method} returned no response")
    if "error" in response:
        raise RuntimeError(response["error"].get("message") or f"{method} failed")
    return response.get("result") or {}


def _run_compress(args: argparse.Namespace) -> dict:
    resumed = _call("session.resume", {"session_id": args.session_key, "cols": 80})
    gateway_session_id = resumed.get("session_id")
    if not gateway_session_id:
        raise RuntimeError("session.resume did not return a gateway session id")
    try:
        return _call(
            "session.compress",
            {
                "session_id": gateway_session_id,
                "focus_topic": args.focus_topic or "",
            },
        )
    finally:
        try:
            _call("session.close", {"session_id": gateway_session_id})
        except Exception:
            pass


def _run_clipboard_save() -> dict:
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as exc:
        raise RuntimeError(f"clipboard unavailable: {exc}") from exc

    image_dir = Path(server._hermes_home) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if save_clipboard_image(image_path):
        return {"path": str(image_path)}
    if has_clipboard_image():
        raise RuntimeError("Clipboard has image but extraction failed")
    raise RuntimeError("No image found in clipboard")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="action", required=True)

    compress = subparsers.add_parser("compress", add_help=False)
    compress.add_argument("--session-key", required=True)
    compress.add_argument("--focus-topic", default="")

    subparsers.add_parser("clipboard-save", add_help=False)

    token = bind_transport(_NullTransport())
    try:
        args = parser.parse_args()
        if args.action == "compress":
            result = _run_compress(args)
        elif args.action == "clipboard-save":
            result = _run_clipboard_save()
        else:
            raise RuntimeError(f"unsupported action: {args.action}")
        payload = {"ok": True, "result": result}
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
    finally:
        reset_transport(token)

    json.dump(payload, sys.__stdout__)
    sys.__stdout__.write("\n")
    sys.__stdout__.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
