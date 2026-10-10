"""Hello-world spike for the network checks (PRD 18.6 / 18.7). Binds 127.0.0.1 only."""

import sys

import uvicorn
from fastapi import FastAPI

HOST = "127.0.0.1"
PORT = 8787

app = FastAPI(title="sair-hello")


@app.get("/")
def hello() -> dict[str, str]:
    return {"hello": "sair"}


def main(host: str = HOST, port: int = PORT) -> None:
    if host != "127.0.0.1":
        sys.exit(f"refusing to bind {host!r}: loopback only (127.0.0.1)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else HOST
    main(host)
