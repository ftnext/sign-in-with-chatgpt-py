"""Loopback callback listener and pasted-URL handling for sign-in.

No storage and no UI framework: the caller injects the warning sink and
the input stream so the CLI and tests can differ.
"""

import os
import secrets
import select
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

from .errors import AuthError

POLL_INTERVAL_SECONDS = 0.1


def callback_query(redirect_uri, text, *, expected_state=None):
    """Turn a pasted callback URL into a query dict, validated for one attempt.

    The URL's scheme, host, port, and path must equal the attempt's redirect
    URI, and every query parameter must appear exactly once. Never includes
    the pasted text in errors: it may carry an authorization code.
    """
    try:
        pasted = urlsplit(text.strip())
        pasted_port = pasted.port
    except ValueError:
        raise AuthError("invalid_callback_url") from None
    expected = urlsplit(redirect_uri)
    if (
        pasted.scheme,
        pasted.hostname,
        pasted_port,
        pasted.path,
    ) != (
        expected.scheme,
        expected.hostname,
        expected.port,
        expected.path,
    ):
        raise AuthError("invalid_callback_url")
    pairs = parse_qsl(pasted.query, keep_blank_values=True)
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise AuthError("invalid_callback_url")
    query = dict(pairs)
    if expected_state is not None and not secrets.compare_digest(
        query.get("state", ""), expected_state
    ):
        raise AuthError("invalid_state")
    if not (query.get("code") or query.get("error")):
        # Neither a grant nor a decline: no usable result exists, so this
        # must not end the attempt (a truncated paste stays recoverable).
        raise AuthError("invalid_callback_url")
    return query


class CallbackListener:
    """One browser authorization result, received on a private loopback listener."""

    def __init__(self, port=0):
        self.result = None
        self.expected_state = None
        self._received = threading.Event()
        self._lock = threading.Lock()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Callback URLs contain a credential: do not log them.

            def do_GET(self):
                host = self.headers.get("Host")
                expected_state = listener.expected_state
                query = None
                if expected_state is not None:
                    try:
                        query = callback_query(
                            listener.redirect_uri,
                            f"http://{host}{self.path}",
                            expected_state=expected_state,
                        )
                    except (AuthError, ValueError):
                        query = None
                body = (
                    "Authorization received. Return to your terminal for the result."
                    if query is not None
                    else "This authorization result could not be verified."
                ).encode()
                self.send_response(200 if query is not None else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)
                if query is not None:
                    listener.submit(query)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.host = f"127.0.0.1:{self.server.server_port}"
        self.redirect_uri = f"http://{self.host}/auth/callback"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self, state):
        self.expected_state = state
        self.thread.start()

    def submit(self, query):
        """Record the attempt's result; only the first caller wins."""
        with self._lock:
            if self.result is not None:
                return False
            self.result = query
            self._received.set()
            return True

    def wait(self, timeout):
        """Wait for the authorization result; returns None on timeout."""
        if self._received.wait(timeout):
            return self.result
        return None

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()


def wait_for_result(listener, deadline, stream=None, warn=None):
    """Wait for the HTTP callback or a pasted callback URL; first wins.

    A pasted URL is only read when ``stream`` is interactive (a TTY); any
    input that fails validation is reported through ``warn`` and the wait
    continues. On non-interactive streams only the HTTP callback is watched.
    No background thread is created. Byte-wise reads stop exactly at the
    newline that completes the pasted URL, so input queued behind it is
    left in the terminal for the shell, not swallowed by sign-in.
    """
    warn = warn or (lambda message: print(message, file=sys.stderr))
    try:
        fd = stream.fileno() if stream is not None else -1
        interactive = fd >= 0 and stream.isatty()
    except (OSError, ValueError, AttributeError):
        interactive = False
    if not interactive:
        return listener.wait(max(0.0, deadline - time.monotonic()))
    pending = bytearray()
    while True:
        result = listener.wait(0)
        if result is not None:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        readable, _, _ = select.select(
            [fd], [], [], min(remaining, POLL_INTERVAL_SECONDS)
        )
        if not readable:
            continue
        byte = os.read(fd, 1)
        if not byte:
            # EOF: only the loopback callback can still complete this
            return listener.wait(max(0.0, deadline - time.monotonic()))
        if byte != b"\n":
            pending += byte
            if len(pending) > 65536:
                warn(
                    "Rejected pasted URL (too long). Paste the full address "
                    "the browser was redirected to."
                )
                pending.clear()
            continue
        text = pending.decode("utf-8", "replace").strip()
        pending.clear()
        if not text:
            continue
        try:
            query = callback_query(
                listener.redirect_uri, text, expected_state=listener.expected_state
            )
        except AuthError as exc:
            warn(
                f"Rejected pasted URL ({exc}). Paste the full address the "
                "browser was redirected to."
            )
            continue
        if listener.submit(query):
            return query
        return listener.wait(0)
