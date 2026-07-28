"""Free-port selection: the probe must see a server that is actually listening."""

from __future__ import annotations

import socket
import sys
from contextlib import closing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from freeport import _has_listener, find_free_port, is_free, main

# Deliberately below the ephemeral range (49152+ on Windows). Probing up there is
# unreliable: a loopback connect whose source port happens to equal its destination
# port completes as a TCP simultaneous open, so an unused port answers itself.
BAND = range(8700, 8900)


def _listen(reuse: bool) -> socket.socket:
    """A listening socket on a concrete free port in BAND."""
    for port in BAND:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if reuse:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            sock.close()
            continue
        sock.listen(1)
        return sock
    pytest.skip(f"no free port in {BAND.start}..{BAND.stop - 1} to test with")


def _free_port() -> int:
    """A port in BAND that nothing is using."""
    with closing(_listen(reuse=False)) as sock:
        return sock.getsockname()[1]


def test_is_free_true_for_unused_port():
    assert is_free(_free_port())


def test_is_free_detects_a_plain_listener():
    with closing(_listen(reuse=False)) as sock:
        assert not is_free(sock.getsockname()[1])


def test_is_free_detects_a_so_reuseaddr_listener():
    with closing(_listen(reuse=True)) as sock:
        assert not is_free(sock.getsockname()[1])


def test_has_listener_is_what_catches_a_served_port():
    """The connect probe is the load-bearing half of ``is_free``.

    Motivation, measured against a real Streamlit on 8501: the old bind-only
    check reported the port FREE while the app was serving on it, so `just serve`
    handed 8501 straight back. A cross-process listener holding SO_REUSEADDR lets
    a second bind succeed on Windows; connecting is the only reliable signal.

    Note this in-process fixture does NOT reproduce that bind behaviour -- a
    same-process listener refuses the second bind -- so the assertions here pin
    the connect probe directly rather than pretending to recreate the hijack.
    """
    with closing(_listen(reuse=True)) as sock:
        assert _has_listener(sock.getsockname()[1])
    assert not _has_listener(_free_port())


def test_find_free_port_skips_past_a_live_listener():
    with closing(_listen(reuse=True)) as sock:
        taken = sock.getsockname()[1]
        assert find_free_port(taken) != taken


def test_find_free_port_returns_the_preferred_port_when_free():
    port = _free_port()
    assert find_free_port(port) == port


def test_main_prints_only_the_port_to_stdout(capsys):
    with closing(_listen(reuse=True)) as sock:
        taken = sock.getsockname()[1]
        assert main([str(taken)]) == 0
    out = capsys.readouterr()
    # stdout is captured by `just serve`, so it must stay a bare parseable port.
    assert int(out.out.strip()) != taken
    assert "in use" in out.err
