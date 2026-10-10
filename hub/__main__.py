"""``python -m hub``: run the hub on ``SAIR_HOST``:``SAIR_PORT`` from settings (NFR-P2).

Settings refuse a non-loopback ``SAIR_HOST``; the launcher then exits non-zero with the fixed
settings message (the rejected value is not echoed) before any server or app is built. The
server's access log is off (it prints query strings).
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from hub.settings import LOOPBACK_HOSTS, MSG_HOST, get_settings

__all__ = ["main"]


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        fields = {".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()}
        if "sair_host" in fields:
            print(f"sair: refusing to start: {MSG_HOST}", file=sys.stderr)
        else:
            print(f"sair: invalid settings: {', '.join(sorted(fields))}", file=sys.stderr)
        return 2
    if settings.sair_host not in LOOPBACK_HOSTS:  # belt and braces; settings already refuse
        print(f"sair: refusing to start: {MSG_HOST}", file=sys.stderr)
        return 2

    import uvicorn

    uvicorn.run(
        "hub.app:app",
        host=settings.sair_host,
        port=settings.sair_port,
        access_log=False,
        proxy_headers=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
