"""Serve the API on a Unix socket only the owner can reach.

    uv run --group website python -m website.backend [--socket PATH]

A TCP port on 127.0.0.1 is open to every user of a shared machine; a socket
inside a 0700 directory is not. The socket path is resolved the same way by the
frontend (``MLE_API_SOCKET``, else ``$XDG_RUNTIME_DIR/mle-trajectories/api.sock``,
else ``~/.cache/mle-trajectories/api.sock``).
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

import uvicorn


def default_socket() -> Path:
    if os.environ.get("MLE_API_SOCKET"):
        return Path(os.environ["MLE_API_SOCKET"])
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) if base and Path(base).is_dir() else Path.home() / ".cache"
    return root / "mle-trajectories" / "api.sock"


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    st = path.stat()
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        sys.exit(f"refusing to serve: {path} is not a private directory owned by you")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--socket", type=Path, default=None)
    ap.add_argument("--reload", action="store_true", help="restart on code changes")
    args = ap.parse_args(argv)
    sock = (args.socket or default_socket()).resolve()
    if len(os.fsencode(sock)) > 100:     # sun_path holds 108 bytes including the NUL
        sys.exit(f"socket path too long for AF_UNIX ({len(os.fsencode(sock))} bytes): {sock}")
    _private_dir(sock.parent)
    if sock.exists() or sock.is_socket():
        sock.unlink()
    print(f"API on unix socket {sock}", file=sys.stderr)
    uvicorn.run("website.backend.app:app", uds=str(sock), reload=args.reload,
                reload_dirs=[str(Path(__file__).parent)] if args.reload else None,
                log_level="info")


if __name__ == "__main__":
    main()
