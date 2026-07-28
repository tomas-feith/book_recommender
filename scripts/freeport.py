"""Pick a free TCP port, preferring a given one.

Streamlit aborts if its port is taken, which happens whenever another local
dashboard is already running on 8501. `just serve` calls this first and passes
the result through as --server.port.

    python scripts/freeport.py        # prints 8501, or the next free port
    python scripts/freeport.py 8600   # start looking at 8600 instead

The chosen port goes to stdout (so a script can capture it); anything human-
facing goes to stderr. Stdlib only, so this runs before any deps are installed.
"""

from __future__ import annotations

import socket
import sys

DEFAULT_PORT = 8501
SEARCH_LIMIT = 50
# Loopback connect timeout. A closed port refuses immediately, so this only
# bounds the pathological case (a firewall dropping packets), never the scan.
PROBE_TIMEOUT = 0.2


def _has_listener(port: int) -> bool:
    """True if something is already accepting connections on this port.

    A bind() test alone is not enough on Windows. Streamlit's listener sets
    SO_REUSEADDR, and Windows then permits a *second* bind to the same address,
    so the probe below would report an actively-served port as free and hand it
    straight back -- the exact case this script exists to prevent.
    SO_EXCLUSIVEADDRUSE on the probe socket does not help either: it stops
    others hijacking *our* socket, not us binding over theirs.

    Connecting is the reliable signal. Both loopback families are tried because
    a server may listen on only one of them (Streamlit binds dual-stack, but a
    stray process may not).
    """
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)), (socket.AF_INET6, ("::1", port))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(PROBE_TIMEOUT)
                if sock.connect_ex(addr) == 0:
                    return True
        except OSError:
            continue  # family unavailable (no IPv6 stack) -- not evidence either way
    return False


def is_free(port: int) -> bool:
    """True if a server can actually take this port.

    Two independent checks, because each misses what the other catches:

    * ``bind`` -- catches ports held exclusively, and ports we lack permission
      for. Done without SO_REUSEADDR on purpose; that option would let us bind
      over a port another process already holds.
    * ``_has_listener`` -- catches an active server whose own SO_REUSEADDR lets
      our bind succeed anyway (see above).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return not _has_listener(port)


def find_free_port(preferred: int = DEFAULT_PORT, limit: int = SEARCH_LIMIT) -> int:
    for port in range(preferred, preferred + limit):
        if is_free(port):
            return port
    raise RuntimeError(f"no free port in {preferred}..{preferred + limit - 1}")


def main(argv: list[str]) -> int:
    preferred = int(argv[0]) if argv else DEFAULT_PORT
    try:
        port = find_free_port(preferred)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if port != preferred:
        print(f">> port {preferred} is in use, using {port} instead", file=sys.stderr)
    print(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
