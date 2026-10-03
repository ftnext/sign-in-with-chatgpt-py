"""Standalone entry point: single-worker loopback server."""

import sys

from pydantic import ValidationError

from .app import create_app
from .config import Settings
from .registrations import StorageError


def main() -> int:
    try:
        settings = Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{error['loc'][0]}: {error['msg']}" for error in exc.errors()
        )
        print(f"Invalid configuration: {problems}", file=sys.stderr)
        return 2
    app = create_app(settings)
    import uvicorn

    print(
        f"Serving on http://{settings.loopback_host} "
        f"(plan inference {'enabled' if settings.chatgpt_plan_enabled else 'disabled'})",
        file=sys.stderr,
    )
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=settings.chatgpt_app_port,
            workers=1,
            access_log=False,
        )
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
