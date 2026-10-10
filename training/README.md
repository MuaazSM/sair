# Sair-Writer export and gates

How the trained LoRA adapter becomes the Q4_K_M GGUF that `llama-server` serves, and how
we decide whether it ships (PRD §12.4, §12.5). Everything here runs on synthetic data only.
The scripts print sizes, hashes, counts, rates and reason codes. They never print model
output, item text or keys.

| Script | What it does |
|---|---|
| `training/export_adapter.py` | `download` → `merge` → `convert` → `quantize`, plus `base` (the same convert and quantize on the untouched base, for baseline C). Each step is idempotent. |
| `training/renderer_runtime.py` | The raw `/completion` prompt string, built exactly as in training (`training/renderer.json`). `renderer` is for D. `few_shot_renderer` is for C. |
| `training/gates.py` | `c-dev` (the base few-shot dev reference), `g-export` and `g-ship`. They write JSON to `training/gates/` and print a recommendation. They never edit `.env` files. |

## Before you start

- The Qwen3.5-4B **bf16 weights** have to be in the HF cache. Right now the cache only holds
  the config and tokenizer. `merge` and `base` refuse without the weights.
  `scripts/models.sh` checks for `config.json` only, so it won't catch this.
  ```bash
  (set -a; . ./.env.training; set +a; unset HF_HUB_OFFLINE; uv run --with huggingface_hub hf download Qwen/Qwen3.5-4B)
  ```
- Disk: about 9 GB for the merged HF dir, 8 GB for each f16 GGUF and 2.7 GB for each Q4_K_M.
  You can delete the f16 files after quantizing, but then the next run rebuilds them.
- Pick the checkpoint `<k>`: the epoch that the §12.3 rule selects from
  `training/runs/<run>.json`. The rule needs the base few-shot reference, so if you don't
  know `<k>` yet, do steps 6 and 9 first (they don't need the adapter). Then run
  `uv run python training/gates.py select --run <run_id>`.

## Run order

```bash
# 0. Freeze the two few-shot examples for baseline C, once, after the data is final
uv run python training/renderer_runtime.py freeze-few-shot
uv run python training/renderer_runtime.py fingerprint      # note the few_shot sha8

# 1. Download the adapter. The Tinker key comes in through this subshell only.
(set -a; . ./.env.training; set +a; \
  uv run python training/export_adapter.py download --from-run <run_id> --epoch <k>)
#    -> models/adapters/<run>_sampler_weights_<name>/  (PEFT: adapter_config.json,
#       adapter_model.safetensors, checkpoint_complete)

# 2. Stop every model server, including Rakshak's, and close big desktop apps
make down
launchctl bootout gui/$(id -u)/com.sair.llama-server 2>/dev/null || true
launchctl bootout gui/$(id -u)/com.sair.ollama 2>/dev/null || true
launchctl bootout gui/$(id -u)/com.sair.hub 2>/dev/null || true   # stops the Kokoro child too
pkill -x ollama; pkill -x llama-server                             # anything left over
uv run python training/export_adapter.py preflight --min-free-gb 10

# 3. Merge (CPU, bf16, HF_HUB_OFFLINE=1) -> models/sair-writer-hf/
uv run python training/export_adapter.py merge --adapter models/adapters/<dir>

# 4. Convert -> models/sair-writer-f16.gguf (convert_hf_to_gguf.py --outtype f16 --no-mtp)
uv run python training/export_adapter.py convert

# 5. Quantize -> models/sair-writer-q4_k_m.gguf (llama-quantize Q4_K_M)
uv run python training/export_adapter.py quantize
#    (steps 3-5 in one go: export_adapter.py all --adapter models/adapters/<dir>)

# 6. Base for baseline C, same path -> models/qwen3.5-4b-base-q4_k_m.gguf
uv run python training/export_adapter.py base

# 7. Restart. G-export needs only the writer llama-server.
llama-server -m models/sair-writer-q4_k_m.gguf --host 127.0.0.1 --port 8091 \
  -c 4096 -ngl 99 -np 1 >/dev/null 2>&1 &

# 8. G-export: Tinker checkpoint vs GGUF on the dev split
uv run python training/gates.py g-export --run <run_id> --epoch <k>
#    -> training/gates/g_export.json ; exit 0 = PASS, 1 = FAIL

# 9. Base few-shot dev reference for the §12.3 rule (base on 8092, writer stopped if memory is tight)
llama-server -m models/qwen3.5-4b-base-q4_k_m.gguf --host 127.0.0.1 --port 8092 \
  -c 4096 -ngl 99 -np 1 >/dev/null 2>&1 &
uv run python training/gates.py c-dev                         # -> training/gates/c_dev.json

# 10. Dev eval for G-ship: C+ and D+, one renderer per run
EVAL_QWEN_BASE_ID=qwen3.5-4b-base-q4_k_m-fs<sha8> uv run python eval/run.py --systems C+ \
  --items training/data/dev.jsonl --renderer training.renderer_runtime:few_shot_renderer \
  --out eval/results/gship-c/
uv run python eval/run.py --systems D+ --items training/data/dev.jsonl \
  --renderer training.renderer_runtime:renderer --out eval/results/gship-d/

# 11. G-ship -> training/gates/g_ship.json + the recommended WRITER_BACKEND
uv run python training/gates.py g-ship \
  --records eval/results/gship-c/items.jsonl eval/results/gship-d/items.jsonl

# 12. Bring the normal stack back
make up      # or: make install-agents
```

## Gate rules

**G-export** passes only if, for each of `unsupported_rate`, `fact_recall`, `fallback_rate` and
`valid_json_no_grammar`, |GGUF − Tinker| ≤ 0.02 (2 percentage points, inclusive). Both sides
run on the same dev items with the same production writer prompt: first attempt only,
grammar off, no regeneration, temperature 0. Both are scored by `training.train.score_dev`.
The Tinker side is the per-epoch dev metrics that `train.py` stored for the selected epoch.
If a metric is missing on one side, the gate fails. If it's missing on both, that metric
is n/a. If the lexicon hash or the writer prompt fingerprint changed since training, the
gate refuses to run.

**G-ship** recommends `llama` only if all four of these hold on dev:

1. G-export passed.
2. unsupported_rate(D+) < unsupported_rate(C+).
3. fact_recall(D+) ≥ fact_recall(C+).
4. template_fallback_rate(D+) ≤ template_fallback_rate(C+).

Otherwise it recommends `gemma` and lists a reason code for each failed condition. It also
reports 95% cluster-bootstrap-by-seed CIs (10,000 resamples) for each system and for each
D+ − C+ difference. `borderline: true` means the CI of the unsupported-rate difference
includes 0. You set `WRITER_BACKEND` yourself. The dev set is never reported.
