"""In-memory state: browser sessions, OAuth transactions, the connection.

Everything here vanishes on restart: OAuth tokens, app sessions, CSRF
state, model cache, and pending sign-in transactions. Only registration
data (see registrations.py) is persisted.
"""

import asyncio
import secrets
import time
from dataclasses import dataclass, field

from .errors import AuthError, PublicError

SESSION_TTL_SECONDS = 24 * 3600
TRANSACTION_TTL_SECONDS = 600
SESSION_COOKIE = "fastapi_chatgpt_plan_session"
CSRF_HEADER = "x-csrf-token"


@dataclass
class Session:
    id: str
    csrf: str
    created_at: float
    expires_at: float
    authenticated: bool
    generation: int | None = None


@dataclass
class OAuthTransaction:
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    client_id: str | None
    session_id: str
    expires_at: float
    generation: int = 0
    public_identity: bool = False


@dataclass
class Connection:
    client_id: str
    issuer: str
    subject: str
    email: str | None
    name: str | None
    id_token: str
    access_token: str
    refresh_token: str | None
    scopes: list
    expires_at: float
    generation: int
    plan_permitted: bool
    needs_reauth: bool = False


@dataclass
class MemoryState:
    sessions: dict = field(default_factory=dict)
    transactions: dict = field(default_factory=dict)
    connection: Connection | None = None
    models_cache: dict | None = None
    active_streams: dict = field(default_factory=dict)
    generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _new_session_id(self) -> str:
        while True:
            session_id = secrets.token_urlsafe(32)
            if session_id not in self.sessions:
                return session_id

    async def create_session(self, *, authenticated=False) -> Session:
        async with self.lock:
            session = Session(
                id=self._new_session_id(),
                csrf=secrets.token_urlsafe(32),
                created_at=time.time(),
                expires_at=time.time() + SESSION_TTL_SECONDS,
                authenticated=authenticated,
                generation=self.generation if authenticated else None,
            )
            self.sessions[session.id] = session
            return session

    async def get_session(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is None or session.expires_at <= time.time():
                self.sessions.pop(session_id, None)
                return None
            if session.authenticated:
                session.expires_at = time.time() + SESSION_TTL_SECONDS
            return session

    async def rotate_session(self, session: Session | None) -> Session:
        """Fresh session ID bound to the just-authenticated browser."""
        async with self.lock:
            if session is not None:
                self.sessions.pop(session.id, None)
            new = Session(
                id=self._new_session_id(),
                csrf=secrets.token_urlsafe(32),
                created_at=time.time(),
                expires_at=time.time() + SESSION_TTL_SECONDS,
                authenticated=True,
                generation=self.generation,
            )
            self.sessions[new.id] = new
            return new

    async def destroy_session(self, session_id: str | None) -> None:
        if session_id:
            async with self.lock:
                self.sessions.pop(session_id, None)

    async def begin_transaction(
        self,
        session_id: str,
        verifier: str,
        redirect_uri: str,
        client_id: str | None,
        *,
        public_identity: bool = False,
    ) -> OAuthTransaction:
        tx = OAuthTransaction(
            state=secrets.token_urlsafe(32),
            nonce=secrets.token_urlsafe(32),
            verifier=verifier,
            redirect_uri=redirect_uri,
            client_id=client_id,
            session_id=session_id,
            expires_at=time.time() + TRANSACTION_TTL_SECONDS,
            generation=self.generation,
            public_identity=public_identity,
        )
        async with self.lock:
            now = time.time()
            self.transactions = {
                key: old
                for key, old in self.transactions.items()
                if old.expires_at > now
            }
            self.transactions[tx.state] = tx
        return tx

    async def consume_transaction(
        self, state: str, session_id: str | None
    ) -> OAuthTransaction | None:
        """Atomically take the transaction; usable exactly once."""
        async with self.lock:
            tx = self.transactions.pop(state, None)
            if (
                tx is None
                or tx.expires_at <= time.time()
                or not session_id
                or not secrets.compare_digest(tx.session_id, session_id)
            ):
                return None
            return tx

    async def set_connection(self, connection: Connection) -> None:
        async with self.lock:
            self.generation += 1
            connection.generation = self.generation
            self.connection = connection
            self.models_cache = None

    async def clear_connection(self) -> None:
        async with self.lock:
            self.generation += 1
            self.connection = None
            self.models_cache = None

    async def finish_sign_in(self, connection, tx, commit):
        """Commit verified registration and browser identity as one transition."""
        async with self.lock:
            session = self.sessions.get(tx.session_id)
            if (
                self.generation != tx.generation
                or session is None
                or session.expires_at <= time.time()
            ):
                raise AuthError("invalid_state")
            commit()
            streams = list(self.active_streams)
            self.active_streams.clear()
            self.generation += 1
            connection.generation = self.generation
            self.connection = connection
            self.models_cache = None
            self.sessions.pop(session.id, None)
            rotated = Session(
                self._new_session_id(),
                secrets.token_urlsafe(32),
                time.time(),
                time.time() + SESSION_TTL_SECONDS,
                True,
                self.generation,
            )
            self.sessions[rotated.id] = rotated
        await self._close_streams(streams)
        return rotated

    async def logout(self, session):
        """Detach only this authenticated generation before any network wait."""
        async with self.lock:
            connection = self.connection
            if (
                not session.authenticated
                or connection is None
                or session.generation != connection.generation
            ):
                raise PublicError("sign_in_required", "Not signed in.", 401)
            generation = connection.generation
            self.generation += 1
            self.connection = None
            self.models_cache = None
            self.sessions = {
                key: value
                for key, value in self.sessions.items()
                if value.generation != generation
            }
            self.transactions = {
                key: value
                for key, value in self.transactions.items()
                if value.generation != generation
            }
            streams = [s for s, g in self.active_streams.items() if g == generation]
            for stream in streams:
                self.active_streams.pop(stream, None)
        await self._close_streams(streams)
        return connection

    def is_current(self, connection) -> bool:
        return (
            self.connection is connection and self.generation == connection.generation
        )

    def register_stream(self, stream, connection) -> bool:
        if not self.is_current(connection):
            return False
        self.active_streams[stream] = connection.generation
        return True

    def unregister_stream(self, stream) -> None:
        self.active_streams.pop(stream, None)

    async def close_streams(self) -> None:
        streams = list(self.active_streams)
        self.active_streams.clear()
        await self._close_streams(streams)

    @staticmethod
    async def _close_streams(streams):
        for stream in streams:
            await stream.aclose()
