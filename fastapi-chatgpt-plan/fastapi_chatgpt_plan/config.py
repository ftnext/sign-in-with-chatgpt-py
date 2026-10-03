"""Environment-based settings, loaded once at startup."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_STATE_DIR = "~/.local/share/fastapi-chatgpt-plan"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    chatgpt_plan_enabled: bool = False
    chatgpt_app_port: int = 8000
    chatgpt_state_dir: str = DEFAULT_STATE_DIR
    chatgpt_identity_client_id: str | None = None

    @field_validator("chatgpt_identity_client_id")
    @classmethod
    def _identity_client(cls, value):
        if value is not None:
            value = value.strip()
            if not value or value == "dynamic_agent_client":
                raise ValueError("Use an issued public identity client ID")
        return value

    @field_validator("chatgpt_app_port")
    @classmethod
    def _port_range(cls, value: int) -> int:
        if not 1024 <= value <= 65535:
            raise ValueError("CHATGPT_APP_PORT must be between 1024 and 65535")
        return value

    @field_validator("chatgpt_state_dir")
    @classmethod
    def _expand_dir(cls, value: str) -> str:
        return str(Path(value).expanduser())

    @property
    def state_dir(self) -> Path:
        return Path(self.chatgpt_state_dir)

    @property
    def loopback_host(self) -> str:
        return f"127.0.0.1:{self.chatgpt_app_port}"

    @property
    def origin(self) -> str:
        return f"http://{self.loopback_host}"

    @property
    def redirect_uri(self) -> str:
        return self.origin + "/auth/callback"
