"""NFR-P6 / PRD §13.2: ``eval/run.py --set real`` runs only with loopback backends.

The real-set root is a tmp dir (monkeypatched); the real ``eval/real/`` is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import run
from eval.backends import StubWriter, TemplateWriter

MARKER = "zqx-velvet-ledger"  # synthetic canary text placed in the fake real items
T = f"we heard a crow near the gate, {MARKER}"
OUT = json.dumps(
    {
        "facts": [
            {
                "span": "we heard a crow near the gate",
                "kind": "sound",
                "subject": "crow",
                "certainty": "sure",
                "place": "near the gate",
            }
        ],
        "mission_result": "not_mentioned",
        "mission_span": None,
        "entry": f"Mom and Dad heard a crow near the gate. {MARKER}",
    }
)


@pytest.fixture
def real_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "real"
    root.mkdir()
    items = [
        {
            "id": f"r{k}",
            "seed": f"r{k}",
            "group": "Mom and Dad",
            "mission": None,
            "transcript": T,
            "reference_transcript": T,
            "gold": {"facts": [], "mission_result": "not_mentioned"},
        }
        for k in range(3)
    ]
    (root / "items.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(run, "REAL_ROOT", root)
    monkeypatch.setattr(run, "REAL_RESULTS", tmp_path / "results-real")
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    return root


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.20:11434",
        "https://api.example.com",
        "http://100.64.0.7:8091",
        "http://127.0.0.1.evil.com:80",
        "ftp://127.0.0.1/",
    ],
)
def test_real_set_refused_with_non_loopback_backend(
    real_root: Path, url: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    real_load = run.load_items
    monkeypatch.setattr(run, "load_items", lambda p: opened.append(p) or real_load(p))
    w = StubWriter([OUT], base_url=url)
    code = run.main(
        ["--set", "real", "--systems", "D+", "--resamples", "50"], writers={"sair_writer": w}
    )
    err = capsys.readouterr().err
    assert code == 3
    assert run.MSG_REAL_NOT_LOOPBACK in err and url not in err
    assert w.calls == [] and opened == []  # refused before any real item is read


def test_real_set_refused_when_one_of_several_backends_is_remote(real_root: Path) -> None:
    writers = {
        "sair_writer": StubWriter([OUT], base_url="http://127.0.0.1:8091"),
        "gemma": StubWriter([OUT], base_url="http://10.0.0.2:11434"),
    }
    assert run.main(["--set", "real", "--systems", "B+,D+"], writers=writers) == 3


def test_real_set_refused_with_sentry_dsn(
    real_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://key@o0.ingest.example.io/1")
    w = StubWriter([OUT], base_url="http://127.0.0.1:8091")
    assert run.main(["--set", "real", "--systems", "D+"], writers={"sair_writer": w}) == 3
    assert run.MSG_REAL_SENTRY in capsys.readouterr().err and w.calls == []


def test_real_set_refused_with_out_dir_outside_real_results(
    real_root: Path, tmp_path: Path
) -> None:
    w = StubWriter([OUT], base_url="http://127.0.0.1:8091")
    code = run.main(
        ["--set", "real", "--systems", "D+", "--out", str(tmp_path / "public")],
        writers={"sair_writer": w},
    )
    assert code == 3 and w.calls == []


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8091", "http://localhost:8091", "http://[::1]:8091", None]
)
def test_real_set_accepted_with_loopback_and_prints_no_item_text(
    real_root: Path, tmp_path: Path, url: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    w = StubWriter([OUT], base_url=url)
    code = run.main(
        ["--set", "real", "--systems", "A,D,D+", "--resamples", "50"],
        writers={"sair_writer": w, "template": TemplateWriter()},
    )
    cap = capsys.readouterr()
    assert code == 0 and len(w.calls) == 6
    for stream in (cap.out, cap.err):
        assert MARKER not in stream and "crow" not in stream and "r0" not in stream
    assert "unsupported_rate" in cap.out and "bootstrap=plain" in cap.out
    out_dir = tmp_path / "results-real" / "latest"
    assert (out_dir / "metrics.json").is_file()
    assert not (out_dir / "items.jsonl").exists()  # no per-item output for the real set
    assert MARKER not in (out_dir / "metrics.json").read_text("utf-8")
    metrics = json.loads((out_dir / "metrics.json").read_text("utf-8"))
    assert metrics["transcription"]["wer"]["point"] == 0.0
    assert metrics["systems"]["D+"]["metrics"]["unsupported_rate"]["method"] == "plain"


def test_real_set_error_reports_type_only(
    real_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise RuntimeError(MARKER)

    monkeypatch.setattr(run, "_evaluate", boom)
    w = StubWriter([OUT], base_url="http://127.0.0.1:8091")
    assert run.main(["--set", "real", "--systems", "D"], writers={"sair_writer": w}) == 1
    cap = capsys.readouterr()
    assert MARKER not in cap.err + cap.out and "RuntimeError" in cap.err


def test_is_loopback_url_table() -> None:
    from eval.config import is_loopback_url

    assert is_loopback_url(None)
    assert is_loopback_url("http://127.0.0.1:11434")
    assert is_loopback_url("http://localhost")
    assert is_loopback_url("http://[::1]:1")
    assert not is_loopback_url("http://0.0.0.0:8091")
    assert not is_loopback_url("http://127.0.0.2:8091")
    assert not is_loopback_url("127.0.0.1:8091")  # no scheme
