import io
import os
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from conftest import (
    DISCOVERY,
    ISSUED_CLIENT_ID,
    SUBJECT,
    drive_browser_callback,
    make_id_token,
)
from cryptography.hazmat.primitives.asymmetric import rsa

from llm_chatgpt_plan import oauth, storage
from llm_chatgpt_plan.errors import ApiError
from llm_chatgpt_plan.oauth import (
    DYNAMIC_CLIENT_ID,
    ISSUER,
    PLAN_SCOPE,
    RESOURCE,
    AuthError,
    CallbackListener,
    OAuth,
    callback_query,
    login_flow,
    wait_for_result,
)


def make_tx(
    state="state-1",
    nonce="nonce-1",
    client_id=None,
    redirect_uri="http://127.0.0.1:1/auth/callback",
):
    return {
        "state": state,
        "nonce": nonce,
        "verifier": "v" * 64,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "expires": time.time() + 600,
    }


def persist_into(store, **extra):
    """A persist_client_id callback writing the pending record inline."""
    return lambda issued: store.save_pending(
        {"client_id": issued, "generation": None, **extra}
    )


class TestBegin:
    def test_dynamic_registration_params(self, auth_server):
        oauth = OAuth(auth_server.client())
        tx, url = oauth.begin("http://127.0.0.1:9/auth/callback", host_id="urn:uuid:h")
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [DYNAMIC_CLIENT_ID]
        assert params["agent_name_hint"] == ["llm-chatgpt-plan"]
        assert params["ext_agent_host_id"] == ["urn:uuid:h"]
        assert params["response_type"] == ["code"]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"][0] != tx["verifier"]
        assert params["state"] == [tx["state"]]
        assert params["nonce"] == [tx["nonce"]]
        assert params["resource"] == [RESOURCE]
        assert PLAN_SCOPE in params["scope"][0]
        assert url.startswith(ISSUER)

    def test_reauth_uses_issued_client_and_hints(self, auth_server):
        oauth = OAuth(auth_server.client())
        prior = {"id_token": "stored-id-token", "email": "user@example.com"}
        _, url = oauth.begin(
            "http://127.0.0.1:9/auth/callback",
            host_id="urn:uuid:h",
            client_id=ISSUED_CLIENT_ID,
            prior=prior,
        )
        params = parse_qs(urlsplit(url).query)
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in params
        assert params["id_token_hint"] == ["stored-id-token"]
        assert params["login_hint"] == ["user@example.com"]

    def test_bad_discovery_rejected(self, rsa_key):
        def handler(request):
            return httpx.Response(200, json={"issuer": "https://evil.example"})

        oauth = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(AuthError):
            oauth.begin("http://127.0.0.1:9/auth/callback", host_id="h")


class TestComplete:
    def query(self, **overrides):
        q = {"state": "state-1", "code": "code-1", "client_id": ISSUED_CLIENT_ID}
        q.update(overrides)
        return q

    def test_success_persists_issued_client_id(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            record = oauth.complete(
                make_tx(nonce="nonce"),
                self.query(),
                persist_client_id=persist_into(store),
            )
            assert record["client_id"] == ISSUED_CLIENT_ID
            assert record["subject"] == SUBJECT
            assert PLAN_SCOPE in record["scopes"]
            # the issued ID went to the pending record, not the connection
            assert store.load_pending()["client_id"] == ISSUED_CLIENT_ID
            assert store.load_credentials() is None

    def test_expired_tx(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            tx = make_tx()
            tx["expires"] = time.time() - 1
            with pytest.raises(AuthError, match="expired_sign_in"):
                oauth.complete(tx, self.query(), persist_into(store))

    def test_wrong_state_rejected(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="invalid_state"),
        ):
            oauth.complete(make_tx(), self.query(state="other"), persist_into(store))

    def test_declined_callback(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="sign_in_declined"),
        ):
            oauth.complete(
                make_tx(),
                self.query(error="access_denied", code=None),
                persist_into(store),
            )
        # declined callbacks must not exchange the code
        assert auth_server.token_requests == []

    def test_dynamic_client_id_in_callback_rejected(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="missing_client_id"),
        ):
            oauth.complete(
                make_tx(),
                self.query(client_id=DYNAMIC_CLIENT_ID),
                persist_into(store),
            )

    def test_client_id_mismatch_on_reauth(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="client_id_mismatch"),
        ):
            oauth.complete(
                make_tx(client_id=ISSUED_CLIENT_ID),
                self.query(client_id="other-client"),
                persist_into(store),
            )

    def test_missing_code(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="missing_code"),
        ):
            oauth.complete(make_tx(), self.query(code=None), persist_into(store))

    def test_id_token_checks(self, auth_server, rsa_key, state_dir):
        cases = {
            "bad nonce": make_id_token(rsa_key, nonce="other"),
            "bad issuer": make_id_token(rsa_key, issuer="https://evil.example"),
            "bad audience": make_id_token(
                rsa_key, client_id=ISSUED_CLIENT_ID, aud="someone-else"
            ),
            "expired": make_id_token(rsa_key, exp=int(time.time()) - 100),
        }
        for token in cases.values():
            auth_server.id_token_override = token
            oauth = OAuth(auth_server.client())
            with (
                storage.locked_store(state_dir) as store,
                pytest.raises(AuthError, match="invalid"),
            ):
                oauth.complete(
                    make_tx(nonce="nonce"), self.query(), persist_into(store)
                )
            auth_server.token_requests.clear()

    def test_wrong_signing_key_rejected(self, auth_server, state_dir, rsa_key):
        other = rsa.generate_private_key(65537, 2048)
        auth_server.id_token_override = make_id_token(other)
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="invalid"),
        ):
            oauth.complete(make_tx(nonce="nonce"), self.query(), persist_into(store))

    def test_subject_mismatch_on_reauth(self, auth_server, state_dir):
        auth_server.id_token_override = make_id_token(
            auth_server.rsa_key, subject="someone-else"
        )
        oauth = OAuth(auth_server.client())
        with (
            storage.locked_store(state_dir) as store,
            pytest.raises(AuthError, match="account_mismatch"),
        ):
            oauth.complete(
                make_tx(client_id=ISSUED_CLIENT_ID, nonce="nonce"),
                self.query(),
                persist_into(store),
                prior_subject=SUBJECT,
            )

    def test_replace_does_not_persist_client_id_before_verify(
        self, auth_server, state_dir
    ):
        auth_server.token_status = 500
        oauth = OAuth(auth_server.client())
        with storage.locked_store(state_dir) as store:
            store.save_credentials({"client_id": "old", "access_token": "x"})
            with pytest.raises(ApiError, match="HTTP 500"):
                oauth.complete(make_tx(), self.query(), persist_client_id=None)
            assert store.load_credentials()["client_id"] == "old"
            assert store.load_pending() is None


class TestRefresh:
    def record(self, **overrides):
        record = {
            "client_id": ISSUED_CLIENT_ID,
            "subject": SUBJECT,
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "scopes": ["openid", PLAN_SCOPE],
            "expires_at": time.time() + 3600,
        }
        record.update(overrides)
        return record

    def test_fresh_token_returned_without_http(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record())
        oauth = OAuth(auth_server.client())
        token = oauth.ensure_access_token(state_dir)
        assert token == "old-access"
        assert auth_server.token_requests == []

    def test_expired_token_refreshed_and_saved(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(expires_at=time.time() + 10))
        oauth = OAuth(auth_server.client())
        token = oauth.ensure_access_token(state_dir)
        assert token == "access-token-1"
        (form,) = auth_server.token_requests
        assert form["grant_type"] == "refresh_token"
        assert form["client_id"] == ISSUED_CLIENT_ID
        assert form["refresh_token"] == "old-refresh"
        assert form["resource"] == RESOURCE
        assert "scope" not in form
        saved = storage.read_credentials(state_dir)
        assert saved["access_token"] == "access-token-1"
        assert saved["refresh_token"] == "refresh-token-1"

    def test_refresh_failure_asks_for_login(self, auth_server, state_dir):
        auth_server.token_status = 400
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(expires_at=time.time() - 5))
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="reauthorization_required"):
            oauth.ensure_access_token(state_dir)

    def test_no_connection(self, auth_server, state_dir):
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="sign_in_required"):
            oauth.ensure_access_token(state_dir)

    def test_missing_plan_scope(self, auth_server, state_dir):
        with storage.locked_store(state_dir) as store:
            store.save_credentials(self.record(scopes=["openid"]))
        oauth = OAuth(auth_server.client())
        with pytest.raises(AuthError, match="plan_permission_required"):
            oauth.ensure_access_token(state_dir)


REDIRECT_URI = "http://127.0.0.1:8123/auth/callback"


class FakeTTY:
    """A pipe-backed stdin stand-in that reports as a terminal."""

    def __init__(self):
        self._read_fd, self._write_fd = os.pipe()

    def isatty(self):
        return True

    def fileno(self):
        return self._read_fd

    def feed(self, text):
        os.write(self._write_fd, (text + "\n").encode())

    def close_write(self):
        os.close(self._write_fd)
        self._write_fd = -1

    def close(self):
        os.close(self._read_fd)
        if self._write_fd >= 0:
            os.close(self._write_fd)


def paste_url(authorize_url, **params):
    """Build the loopback callback URL for the attempt in authorize_url."""
    parsed = parse_qs(urlsplit(authorize_url).query)
    query = {"state": parsed["state"][0], **params}
    return parsed["redirect_uri"][0] + "?" + urlencode(query), parsed["nonce"][0]


class TestManualUrl:
    def test_hints_removed_from_shown_url(self, auth_server):
        o = OAuth(auth_server.client())
        prior = {"id_token": "stored-id-token", "email": "user@example.com"}
        tx, url = o.begin(
            REDIRECT_URI,
            host_id="urn:uuid:h",
            client_id=ISSUED_CLIENT_ID,
            prior=prior,
        )
        assert "id_token_hint" in url
        shown = o.manual_url(tx)
        params = parse_qs(urlsplit(shown).query)
        assert "id_token_hint" not in params
        assert "login_hint" not in params
        assert params["client_id"] == [ISSUED_CLIENT_ID]
        assert params["state"] == [tx["state"]]
        assert "stored-id-token" not in shown
        assert "user@example.com" not in shown


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
            "https://127.0.0.1:8123/auth/callback?state=s",
            "http://localhost:8123/auth/callback?state=s",
            "http://127.0.0.1:9999/auth/callback?state=s",
            "http://127.0.0.1:8123/other?state=s",
            "http://127.0.0.1/auth/callback?state=s",
            "not a url",
            "javascript:alert(1)",
        ],
    )
    def test_wrong_target_rejected(self, url):
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, url)

    def test_duplicate_params_rejected(self):
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, REDIRECT_URI + "?state=s&state=t")

    def test_wrong_state_rejected(self):
        with pytest.raises(AuthError, match="invalid_state"):
            callback_query(
                REDIRECT_URI,
                REDIRECT_URI + "?state=other&code=c",
                expected_state="s",
            )

    @pytest.mark.parametrize(
        "query",
        [
            "state=s",  # truncated paste: no code, no error
            "state=s&foo=bar",
            "state=s&code=",
        ],
    )
    def test_no_result_params_rejected(self, query):
        # a callback with neither code nor error must not end the attempt
        with pytest.raises(AuthError, match="invalid_callback_url"):
            callback_query(REDIRECT_URI, REDIRECT_URI + "?" + query, expected_state="s")

    def test_error_callback_validates_state(self):
        # an error callback with a foreign state is still rejected
        with pytest.raises(AuthError, match="invalid_state"):
            callback_query(
                REDIRECT_URI,
                REDIRECT_URI + "?state=other&error=access_denied",
                expected_state="s",
            )


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
            assert result == {
                "state": "state-1",
                "code": "c1",
                "client_id": "i1",
            }
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
            tty.feed("http://example.com/not-the-callback")
            # a truncated paste (state but no code) must not end the wait
            tty.feed(listener.redirect_uri + "?state=state-1")
            tty.feed(listener.redirect_uri + "?state=state-1&code=c3&client_id=i3")
            result = wait_for_result(
                listener, time.monotonic() + 5, stream=tty, warn=warnings.append
            )
            assert result["code"] == "c3"
            assert len(warnings) == 2
            assert "Rejected" in warnings[0]
            # the rejected URL is never echoed into the warning
            assert "example.com" not in warnings[0]
        finally:
            listener.close()
            tty.close()

    def test_input_after_callback_line_is_left_in_stream(self):
        # input queued behind the pasted URL belongs to the shell, not us
        listener = self.listener()
        tty = FakeTTY()
        try:
            tty.feed(listener.redirect_uri + "?state=state-1&code=c&client_id=i")
            tty.feed("echo leftover")
            result = wait_for_result(
                listener, time.monotonic() + 5, stream=tty, warn=lambda m: None
            )
            assert result["code"] == "c"
            assert os.read(tty._read_fd, 100) == b"echo leftover\n"
        finally:
            listener.close()
            tty.close()

    def test_non_tty_stream_is_never_read(self):
        listener = self.listener()
        try:
            stream = io.StringIO(
                listener.redirect_uri + "?state=state-1&code=c&client_id=i\n"
            )
            result = wait_for_result(listener, time.monotonic() + 0.2, stream=stream)
            assert result is None  # timed out: paste path stays off
            assert stream.getvalue().endswith("\n")  # nothing consumed
        finally:
            listener.close()

    def test_eof_falls_back_to_listener(self):
        listener = self.listener()
        tty = FakeTTY()
        try:
            tty.close_write()
            thread = threading.Thread(
                target=lambda: httpx.get(
                    listener.redirect_uri + "?state=state-1&code=c4&client_id=i4"
                ),
                daemon=True,
            )
            thread.start()
            result = wait_for_result(listener, time.monotonic() + 10, stream=tty)
            thread.join()
            assert result["code"] == "c4"
        finally:
            listener.close()
            tty.close()

    def test_paste_losing_race_returns_listener_result(self, monkeypatch):
        # if the HTTP result lands while the paste is validated, it wins
        listener = self.listener()
        tty = FakeTTY()
        real = oauth.callback_query

        def query_then_submit(*args, **kwargs):
            query = real(*args, **kwargs)
            listener.submit({"state": "state-1", "via": "http"})
            return query

        monkeypatch.setattr(oauth, "callback_query", query_then_submit)
        try:
            tty.feed(listener.redirect_uri + "?state=state-1&code=c5&client_id=i5")
            result = wait_for_result(
                listener, time.monotonic() + 5, stream=tty, warn=lambda m: None
            )
            assert result == {"state": "state-1", "via": "http"}
        finally:
            listener.close()
            tty.close()


class TestLoginFlow:
    def test_manual_login_via_pasted_url(self, auth_server, state_dir):
        tty = FakeTTY()
        shown = []

        def show(url, browser_failed):
            shown.append(browser_failed)
            paste, nonce = paste_url(url, code="c1", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            record = login_flow(
                state_dir,
                auth_server.client(),
                manual=True,
                timeout=10,
                show_url=show,
                input_stream=tty,
                open_browser=lambda url: pytest.fail("browser used"),
            )
        finally:
            tty.close()
        assert shown == [False]
        assert record["client_id"] == ISSUED_CLIENT_ID
        saved = storage.read_credentials(state_dir)
        assert saved["access_token"] == "access-token-1"
        assert saved["generation"] == 1
        (form,) = auth_server.token_requests
        assert form["code"] == "c1"

    def test_browser_failure_falls_back_to_manual(self, auth_server, state_dir):
        tty = FakeTTY()
        shown = []

        def show(url, browser_failed):
            shown.append(browser_failed)
            paste, nonce = paste_url(url, code="c2", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            record = login_flow(
                state_dir,
                auth_server.client(),
                timeout=10,
                open_browser=lambda url: False,
                show_url=show,
                input_stream=tty,
            )
        finally:
            tty.close()
        assert shown == [True]
        assert record["client_id"] == ISSUED_CLIENT_ID

    def test_paste_and_http_race_exchanges_once(self, auth_server, state_dir):
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c-paste", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)
            parsed = parse_qs(urlsplit(url).query)
            http_callback = (
                parsed["redirect_uri"][0]
                + "?"
                + urlencode(
                    {
                        "state": parsed["state"][0],
                        "code": "c-http",
                        "client_id": ISSUED_CLIENT_ID,
                    }
                )
            )

            def drive():
                time.sleep(0.3)  # let the paste win deterministically
                try:
                    httpx.get(http_callback)
                except httpx.HTTPError:
                    pass  # listener may already be closed

            threading.Thread(target=drive, daemon=True).start()

        try:
            record = login_flow(
                state_dir,
                auth_server.client(),
                manual=True,
                timeout=10,
                show_url=show,
                input_stream=tty,
            )
        finally:
            tty.close()
        assert record["client_id"] == ISSUED_CLIENT_ID
        # exactly one code exchange happened for the attempt
        (form,) = auth_server.token_requests
        assert form["code"] == "c-paste"

    def test_stale_attempt_aborted_by_generation_bump(self, auth_server, state_dir):
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c3", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            # a concurrent logout bumps the generation mid-attempt
            with storage.locked_store(state_dir) as store:
                store.bump_generation()
            tty.feed(paste)

        try:
            with pytest.raises(AuthError, match="connection_changed"):
                login_flow(
                    state_dir,
                    auth_server.client(),
                    manual=True,
                    timeout=10,
                    show_url=show,
                    input_stream=tty,
                )
        finally:
            tty.close()
        # the logged-out state is not resurrected by the stale attempt
        saved = storage.read_credentials(state_dir)
        assert saved == {"generation": 1}

    def test_refresh_during_wait_proves_runtime_lock_free(
        self, auth_server, stored_connection, state_dir, monkeypatch
    ):
        storage.Store(state_dir).save_credentials(
            {**stored_connection, "expires_at": time.time() - 5}
        )
        refreshed = []
        real = oauth.callback_query

        def query_then_refresh(*args, **kwargs):
            query = real(*args, **kwargs)
            # mid-wait a refresh must still take the runtime lock
            refreshed.append(OAuth(auth_server.client()).ensure_access_token(state_dir))
            return query

        monkeypatch.setattr(oauth, "callback_query", query_then_refresh)
        tty = FakeTTY()

        def show(url, browser_failed):
            paste, nonce = paste_url(url, code="c4", client_id=ISSUED_CLIENT_ID)
            auth_server.nonce_holder["nonce"] = nonce
            tty.feed(paste)

        try:
            login_flow(
                state_dir,
                auth_server.client(),
                manual=True,
                timeout=10,
                show_url=show,
                input_stream=tty,
            )
        finally:
            tty.close()
        assert refreshed == ["access-token-1"]
        (refresh_form, exchange_form) = auth_server.token_requests
        assert refresh_form["grant_type"] == "refresh_token"
        assert exchange_form["grant_type"] == "authorization_code"
        # refresh did not bump the generation; the new login did
        assert storage.read_credentials(state_dir)["generation"] == 2

    def test_failed_exchange_leaves_pending_for_retry(self, auth_server, state_dir):
        auth_server.token_status = 500

        def drive(url):
            drive_browser_callback(url, nonce_holder=auth_server.nonce_holder)
            return True

        with pytest.raises(ApiError, match="HTTP 500"):
            login_flow(
                state_dir,
                auth_server.client(),
                timeout=10,
                open_browser=drive,
            )
        pending = storage.Store(state_dir).load_pending()
        assert pending == {"client_id": ISSUED_CLIENT_ID, "generation": None}
        assert storage.read_credentials(state_dir) is None

        # the retry reuses the issued client id instead of re-registering
        auth_server.token_status = 200
        seen = []

        def drive2(url):
            seen.append(parse_qs(urlsplit(url).query))
            drive_browser_callback(url, nonce_holder=auth_server.nonce_holder)
            return True

        record = login_flow(
            state_dir,
            auth_server.client(),
            timeout=10,
            open_browser=drive2,
        )
        assert seen[0]["client_id"] == [ISSUED_CLIENT_ID]
        assert "agent_name_hint" not in seen[0]
        assert record["client_id"] == ISSUED_CLIENT_ID
        assert storage.Store(state_dir).load_pending() is None

    def test_pending_ignored_for_replace(
        self, auth_server, stored_connection, state_dir
    ):
        storage.Store(state_dir).save_pending(
            {"client_id": "unrelated-client", "generation": 0}
        )
        seen = []

        def drive(url):
            seen.append(parse_qs(urlsplit(url).query))
            drive_browser_callback(
                url, client_id="new-client", nonce_holder=auth_server.nonce_holder
            )
            return True

        record = login_flow(
            state_dir,
            auth_server.client(),
            replace=True,
            timeout=10,
            open_browser=drive,
        )
        assert seen[0]["client_id"] == [DYNAMIC_CLIENT_ID]
        assert "agent_name_hint" in seen[0]
        assert record["client_id"] == "new-client"
        assert storage.Store(state_dir).load_pending() is None

    def test_pending_from_other_generation_deleted(self, auth_server, state_dir):
        storage.Store(state_dir).save_pending(
            {"client_id": "stale-client", "generation": 7}
        )
        seen = []

        def drive(url):
            seen.append(parse_qs(urlsplit(url).query))
            drive_browser_callback(url, nonce_holder=auth_server.nonce_holder)
            return True

        record = login_flow(
            state_dir,
            auth_server.client(),
            timeout=10,
            open_browser=drive,
        )
        # the unrelated registration was discarded, not reused
        assert seen[0]["client_id"] == [DYNAMIC_CLIENT_ID]
        assert record["client_id"] == ISSUED_CLIENT_ID
        assert storage.Store(state_dir).load_pending() is None


class TestRevokeRefreshToken:
    def test_posts_refresh_token_and_client_id(self, auth_server):
        o = OAuth(auth_server.client())
        assert o.revoke_refresh_token("client-1", "rt-1", sleep=lambda s: None)
        (form,) = auth_server.revoke_requests
        assert form == {
            "token": "rt-1",
            "token_type_hint": "refresh_token",
            "client_id": "client-1",
        }

    def test_client_error_not_retried(self, auth_server):
        auth_server.revoke_statuses = [403]
        o = OAuth(auth_server.client())
        assert not o.revoke_refresh_token("c", "r", sleep=lambda s: None)
        assert len(auth_server.revoke_requests) == 1

    def test_server_errors_retried_then_success(self, auth_server):
        auth_server.revoke_statuses = [500, 502, 200]
        o = OAuth(auth_server.client())
        assert o.revoke_refresh_token("c", "r", sleep=lambda s: None)
        assert len(auth_server.revoke_requests) == 3

    def test_retry_is_bounded(self, auth_server):
        auth_server.revoke_statuses = [500] * 10
        o = OAuth(auth_server.client())
        assert not o.revoke_refresh_token("c", "r", attempts=2, sleep=lambda s: None)
        assert len(auth_server.revoke_requests) == 2

    def test_missing_endpoint_returns_false(self, auth_server):
        def handler(request):
            if str(request.url).endswith("openid-configuration"):
                return httpx.Response(
                    200,
                    json={
                        k: v for k, v in DISCOVERY.items() if k != "revocation_endpoint"
                    },
                )
            return auth_server.handler(request)

        o = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        assert not o.revoke_refresh_token("c", "r", sleep=lambda s: None)
        assert auth_server.revoke_requests == []

    def test_foreign_revocation_endpoint_rejected(self, auth_server):
        def handler(request):
            if str(request.url).endswith("openid-configuration"):
                bad = dict(
                    DISCOVERY,
                    revocation_endpoint="https://evil.example/revoke",
                )
                return httpx.Response(200, json=bad)
            return auth_server.handler(request)

        o = OAuth(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(AuthError, match="invalid_discovery"):
            o.metadata()
