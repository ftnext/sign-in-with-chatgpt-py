import io
import os
import threading
import time

import httpx
import pytest
from conftest import FakeTTY

from chatgpt_auth.callback import (
    CallbackListener,
    callback_query,
    wait_for_result,
)
from chatgpt_auth.errors import AuthError

REDIRECT_URI = "http://127.0.0.1:9/auth/callback"


class TestCallbackQuery:
    def test_valid_url(self):
        query = callback_query(
            REDIRECT_URI,
            REDIRECT_URI + "?state=s&code=c&client_id=i",
            expected_state="s",
        )
        assert query == {"state": "s", "code": "c", "client_id": "i"}

    def test_error_callback_passes_through(self):
        query = callback_query(
            REDIRECT_URI,
            REDIRECT_URI + "?state=s&error=access_denied",
            expected_state="s",
        )
        assert query["error"] == "access_denied"

    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1:9/auth/callback?state=s&code=c",
            "http://localhost:9/auth/callback?state=s&code=c",
            "http://127.0.0.1:10/auth/callback?state=s&code=c",
            "http://127.0.0.1:9/other?state=s&code=c",
        ],
    )
    def test_wrong_target_rejected(self, url):
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, url)

    def test_duplicate_params_rejected(self):
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, REDIRECT_URI + "?state=a&state=b&code=c")

    def test_wrong_state_rejected(self):
        with pytest.raises(AuthError, match="invalid_state"):
            callback_query(
                REDIRECT_URI,
                REDIRECT_URI + "?state=other&code=c",
                expected_state="s",
            )

    def test_error_callback_validates_state(self):
        with pytest.raises(AuthError, match="invalid_state"):
            callback_query(
                REDIRECT_URI,
                REDIRECT_URI + "?state=other&error=access_denied",
                expected_state="s",
            )

    @pytest.mark.parametrize("query", ["state=s", "foo=bar", "state=s&foo=1"])
    def test_no_result_params_rejected(self, query):
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, REDIRECT_URI + "?" + query)

    def test_malformed_url_rejected(self):
        with pytest.raises(AuthError):
            callback_query(REDIRECT_URI, "http://127.0.0.1:9a/path?state=s&code=c")


class TestListener:
    def test_first_result_wins(self):
        listener = CallbackListener(0)
        listener.start("state-1")
        try:
            assert listener.submit({"a": 1}) is True
            assert listener.submit({"b": 2}) is False
            assert listener.result == {"a": 1}
        finally:
            listener.close()

    def test_invalid_callback_does_not_end_attempt(self):
        listener = CallbackListener(0)
        listener.start("state-1")
        try:
            bad = httpx.get(listener.redirect_uri + "?state=wrong&code=c")
            assert bad.status_code == 400
            good = httpx.get(listener.redirect_uri + "?state=state-1&code=c")
            assert good.status_code == 200
            assert listener.wait(2)["code"] == "c"
        finally:
            listener.close()

    def test_wait_returns_none_on_timeout(self):
        listener = CallbackListener(0)
        listener.start("state-1")
        try:
            assert listener.wait(0.2) is None
        finally:
            listener.close()


class TestWaitForResult:
    def listener(self):
        listener = CallbackListener(0)
        listener.start("state-1")
        return listener

    def test_pasted_url_completes(self):
        listener = self.listener()
        tty = FakeTTY()
        try:
            tty.feed(listener.redirect_uri + "?state=state-1&code=c1&client_id=i1")
            result = wait_for_result(
                listener, time.monotonic() + 5, stream=tty, warn=lambda m: None
            )
            assert result == {"state": "state-1", "code": "c1", "client_id": "i1"}
        finally:
            listener.close()
            tty.close()

    def test_http_callback_completes_while_tty_waits(self):
        listener = self.listener()
        tty = FakeTTY()
        try:
            thread = threading.Thread(
                target=lambda: httpx.get(
                    listener.redirect_uri + "?state=state-1&code=c2&client_id=i2"
                ),
                daemon=True,
            )
            thread.start()
            result = wait_for_result(listener, time.monotonic() + 10, stream=tty)
            thread.join()
            assert result["code"] == "c2"
        finally:
            listener.close()
            tty.close()

    def test_invalid_paste_warns_and_keeps_waiting(self):
        listener = self.listener()
        tty = FakeTTY()
        warnings = []
        try:
            tty.feed("http://evil.example/cb?state=state-1&code=bad")
            tty.feed(listener.redirect_uri + "?state=state-1&code=good")
            result = wait_for_result(
                listener,
                time.monotonic() + 5,
                stream=tty,
                warn=warnings.append,
            )
            assert result["code"] == "good"
            assert warnings and "Rejected pasted URL" in warnings[0]
        finally:
            listener.close()
            tty.close()

    def test_input_after_callback_line_is_left_in_stream(self):
        listener = self.listener()
        tty = FakeTTY()
        try:
            os.write(
                tty._write_fd,
                (
                    listener.redirect_uri + "?state=state-1&code=c\nnext-shell-input\n"
                ).encode(),
            )
            result = wait_for_result(
                listener, time.monotonic() + 5, stream=tty, warn=lambda m: None
            )
            assert result["code"] == "c"
            leftover = os.read(tty._read_fd, 1024)
            assert leftover == b"next-shell-input\n"
        finally:
            listener.close()
            tty.close()

    def test_non_tty_stream_is_never_read(self):
        listener = self.listener()
        stream = io.BytesIO(b"http://x/?state=state-1&code=c\n")
        try:
            result = wait_for_result(listener, time.monotonic() + 0.2, stream=stream)
            assert result is None
        finally:
            listener.close()

    def test_eof_falls_back_to_listener(self):
        listener = self.listener()
        tty = FakeTTY()
        try:
            tty.close_write()
            thread = threading.Thread(
                target=lambda: (
                    time.sleep(0.2),
                    httpx.get(listener.redirect_uri + "?state=state-1&code=c5"),
                ),
                daemon=True,
            )
            thread.start()
            result = wait_for_result(
                listener, time.monotonic() + 10, stream=tty, warn=lambda m: None
            )
            thread.join()
            assert result["code"] == "c5"
        finally:
            listener.close()
            tty.close()
