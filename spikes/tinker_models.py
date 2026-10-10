"""P0 Tinker spike (PRD 18.10): base models, candidate IDs, usage, adapter download path.

    (set -a; . ./.env.training; set +a; uv run python spikes/tinker_models.py)

Read-only calls (no training, no sampling, no credits). Never prints the key or the environment;
errors are reported as exception type + HTTP status only.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime, timedelta


def _err(e: BaseException) -> str:
    return f"{type(e).__name__} status={getattr(e, 'status_code', getattr(e, 'status', '?'))}"


def _size_b(name: str) -> float:
    """Total params in billions parsed from a model name ('A3B' active counts excluded)."""
    m = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)[Bb](?![A-Za-z])", name.replace("-A", " A"))
    nums = [float(x) for x in m]
    return max(nums) if nums else 0.0


def main() -> None:
    import tinker

    print(f"tinker sdk {tinker.__version__ if hasattr(tinker, '__version__') else '?'}")
    try:
        sc = tinker.ServiceClient()
        caps = sc.get_server_capabilities()
    except Exception as e:  # noqa: BLE001
        print(f"get_server_capabilities failed: {_err(e)}")
        sys.exit(1)
    models = caps.supported_models
    print(f"supported_models: {len(models)}")
    print(f"{'model_name':52} {'ctx':>7} {'train':>5} {'sample':>6}")
    for m in sorted(models, key=lambda m: m.model_name):
        print(
            f"{m.model_name:52} {str(m.max_context_length):>7} {str(m.trainable):>5} "
            f"{str(m.sampleable):>6}"
        )
    names = [m.model_name for m in models]
    base = [n for n in names if re.search(r"qwen3\.5-4b", n, re.I)]
    print(f"\ncandidates TINKER_BASE_MODEL (Qwen3.5-4B): {base}")
    instruct = [
        m
        for m in models
        if m.sampleable is not False and not re.search(r"base|-pt\b", m.model_name, re.I)
    ]
    instruct.sort(key=lambda m: _size_b(m.model_name), reverse=True)
    print("largest sampleable non-base models (teacher candidates):")
    for m in instruct[:6]:
        print(f"  {m.model_name}  (~{_size_b(m.model_name):g}B by name)")

    try:
        rc = sc.create_rest_client()
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)  # hour-aligned
        usage = rc.get_billing_usage(end - timedelta(days=14), end).result()  # max 14 d
        cost = sum(
            getattr(ev, "estimated_cost_usd", None)
            or getattr(getattr(ev, "event_info", None), "estimated_cost_usd", None)
            or 0.0
            for ev in usage.data
        )
        print(
            f"\nbilling usage last 14 d: events={len(usage.data)} "
            f"estimated_gross_cost_usd={cost:.4f} (before credits; not a balance)"
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nbilling usage: unavailable ({_err(e)})")
    print(
        "credit balance: no balance endpoint in tinker SDK "
        f"{getattr(tinker, '__version__', '?')} (RestClient has get_billing_usage only); "
        "see the console"
    )
    try:
        print(f"console: {sc.get_console_url()}")
    except Exception as e:  # noqa: BLE001
        print(f"console url: unavailable ({_err(e)})")
    print("\nadapter download path (documented in the SDK/CLI):")
    print("  CLI : tinker checkpoint download tinker://<run-id>/weights/<name> --output models/")
    print(
        "  SDK : ServiceClient().create_rest_client()"
        ".get_checkpoint_archive_url_from_tinker_path('tinker://<run-id>/weights/<name>')"
        " -> signed URL to a .tar archive (PEFT LoRA adapter)"
    )
    print("  also: tinker checkpoint push-hf (not used: NFR-P6, keep adapters private/local)")


if __name__ == "__main__":
    main()
