# AI Video Generator — Project Flow

Offline text-to-video CLI built on `diffusers`. A prompt comes in on the command line, the best model is selected and downloaded if needed, then the diffusion pipeline generates frames that are assembled into an MP4.

## Run

```bash
python main.py "a 480p sunset timelapse, 10 seconds"
python main.py "a cat playing piano" --duration 10 --resolution 1080 --seed 42
python main.py --benchmark            # measure model paths, no video written
python main.py --list-models          # show available models
python main.py --summary              # print hardware config
```

## What the user provides

A free-text prompt plus optional overrides:

| Flag | Meaning |
|---|---|
| `--model` | `cogvideox-5b`, `cogvideox-2b`, `ltx-video`, `ltx-video-distilled` (auto-selected if omitted) |
| `--duration` | seconds of output (1–60) |
| `--fps` | frames per second |
| `--resolution` | `480` / `720` / `1080` |
| `--seed` | reproducibility |
| `--steps` | denoising steps |
| `--guidance` | classifier-free guidance scale |
| `--output` | output `.mp4` path |
| `--fast` | distilled LTX preset: 20 steps, guidance 3.0, 480p @ 24 fps, streaming |
| `--stream` | write and play each pass as soon as it finishes |
| `--segment-secs` | cap a single pass to N seconds of footage |
| `--preview-every` | decode a live preview still every N denoising steps |
| `--no-compile` | disable `torch.compile` |
| `--cache-dir` | where finished passes are cached (default `./.gen_cache`) |
| `--no-resume` | ignore cached passes and regenerate everything |
| `--clear-cache` | delete cached runs for this cache dir, then continue |
| `--idle-timeout` | seconds of inactivity before unloading model (default 180) |

Prompt text itself can also carry hints: `"4k"`, `"1080p"`, `"720p"`, `"480p"`, `"sd"`, `"10 seconds"`, `"24fps"`, `"high quality"`, `"draft"`. CLI flags still win over prompt hints.

## Module responsibilities

### `hardware.py`

`detect_device()` inspects the environment and returns a config dict:

- **CUDA** — device `"cuda"`, dtype chosen by compute capability (`bfloat16` on Ampere+, `float16` on Turing/T4), VRAM per GPU, total VRAM, gpu count, whether `device_map="balanced"` sharding is used, whether `torch.compile` is safe, max resolution tier.
- **MPS** — Apple Silicon.
- **CPU** — sets thread counts for OpenMP/MKL/OpenBLAS and falls back to `bfloat16` with sequential CPU offload.

`print_device_summary()` formats that dict for the user.

### `model.py`

Contains the model registry (`MODELS`), auto-selection (`select_model()`), version guards, and the download + load path.

- **Selection** picks the safest model for the detected hardware: `cogvideox-5b` only on a single ≥24 GB GPU; `cogvideox-2b` on ≥10 GB; `ltx-video` below that, on CPU, or on MPS.
- **Download** (`ensure_model_downloaded`) uses an explicit per-repo file list (`_REQUIRED_FILES`) with `hf_hub_download()` one file at a time, so LTX-Video only pulls the ~8 GB text-to-video weights instead of the whole repo. A snapshot directory is populated and verified by required-file presence so a stale cache does not trigger a full re-download.
- **Load** (`load_pipeline` → `_load_pipeline`) calls `from_pretrained()` with `device_map="balanced"` when multiple GPUs are present, so weights are sharded across all of them instead of streamed through one GPU over PCIe. On sharded-load failure it falls back to single-GPU residency + CPU offload.
- **Optimisations** (`_apply_optimisations`) fix attention before placement: try `xformers` first, fall back to the chunked attention patch, then place the pipeline (fully resident, model CPU offload, or sequential CPU offload depending on VRAM). VAE slicing/tiling is always enabled. `torch.compile` is applied only on a single fully-resident GPU.
- **Distilled LTX** (`_swap_ltx_distilled_transformer`) hot-swaps the base transformer for the distilled checkpoint after the base pipeline loads.

### `chunked_attention.py`

CogVideoX computes joint text+video attention, so a 480p/49-frame clip has a score matrix on the order of tens of GB per block — too large for a T4 even with sharding. This module patches the existing `CogVideoXAttnProcessor` and only replaces the `F.scaled_dot_product_attention` call with a Q-chunked equivalent that never materialises the full N² matrix. Peak score memory becomes `O(chunk × heads × N)` instead of `O(heads × N²)`. It delegates to the original processor so it stays correct across diffusers versions (tuple vs. single-tensor return shapes, 3D rotary embedding forwarding, etc.).

### `generator.py`

Drives the actual generation and writes MP4s.

Workflow inside `generate_video()`:

1. **Parse prompt hints** — resolution, duration, fps, quality multiplier.
2. **Resolve parameters** — CLI flag > prompt hint > model default > hard default (480p, 10 s, model fps).
3. **Frame planning** — compute the target output frame count (`duration × fps`), the per-pass native frame cap (49 for CogVideoX, ~121 for LTX), and split into segments if streaming. A VRAM safety heuristic can cap frames per pass on CUDA when activations would OOM.
4. **Prompt preparation** — enhance the prompt with a quality suffix, build a negative prompt, optionally build a live-preview hook.
5. **Cache + embeddings** — construct a `RunSignature`, register with `RunCache`, and encode the prompt once (or load cached T5 embeddings) so every pass of the run reuses the same text encoding.
6. **Per-pass generation** — for each segment, either reuse a cached pass or run the pipeline with OOM recovery (one retry at lower resolution and fewer frames after a CUDA OOM). Each finished pass is saved losslessly to disk as `.npz`.
7. **Streaming** — when enabled and there are multiple passes, each pass is encoded and shown as soon as it finishes.
8. **Assembly** — cross-dissolve segments together, retime to the requested duration with blend interpolation if the generated frame count differs from the target, then write the final MP4.
9. **Export** — write MP4 via `imageio` (libx264, yuv420p), falling back to OpenCV.

The pipeline call itself (`_run_pipeline`) shares one code path for CogVideoX and LTX, passes cached embeddings when available, and falls back to the pipeline's own encoding if the cached embeddings are rejected — but never swallows a CUDA OOM, which is handled by `_run_with_oom_recovery`.

### `cache.py`

Crash-resilient pass cache.

- A run is identified by a `RunSignature` hash of prompt, negative prompt, model, resolution, fps, steps, guidance, seed, duration, and per-pass frame counts. Any difference produces a different cache key, so two configurations can never splice into one file.
- Layout under `<cache_root>/<run_key>/`: `run.json`, `pass_01.npz`, …, `prompt_embeds.npz`, `done.json`.
- Passes are written to a temporary file and renamed, so an interrupted save never leaves a half-pass. Loading a pass validates shape before reuse.
- Prompt embeddings are cached as raw T5 output (fp32 storage with the original dtype recorded) so a resumed run can skip text encoding entirely.
- Granularity is one pass. There is no mid-pass checkpoint — resuming inside a single denoising loop would require re-implementing the scheduler loop that `pipe.__call__` owns.

### `idle_guard.py`

Background watchdog thread that unloads the pipeline after `idle_seconds` of inactivity. `ping()` resets the timer around generation; `get_pipe()` transparently reloads from cache if the pipeline was already unloaded. `stop()` is registered with `atexit` and also callable on Ctrl+C.

### `benchmark.py`

Loads each requested model, probes a small number of denoising steps plus one VAE decode, records seconds per step (median, with the first step reported separately), decode time, and peak VRAM per GPU, then projects each path to realistic clip lengths using the same segment planner as generation. No video is written.

## End-to-end flow

1. `main()` parses args, applies the `--fast` preset if requested, calls `hardware.detect_device()`, prints the hardware summary.
2. If `--benchmark`, it runs `benchmark.run_benchmark()` and exits.
3. Otherwise it calls `model.load_pipeline()`, which downloads (if needed) and loads the selected pipeline with the appropriate offload/sharding strategy and attention fix.
4. If `--idle-timeout` is set, `IdleGuard` starts and `atexit` is wired to stop it.
5. `generator.generate_video()` is called with the resolved parameters. It parses prompt hints, plans segments, prepares the prompt and embeddings, then runs each pass with OOM recovery.
6. Each finished pass is cached; if streaming, each pass is written and shown immediately.
7. Segments are cross-dissolved, the clip is retimed to the requested duration if needed, and the final MP4 is written and shown.
8. If the idle guard is active, the process stays alive so another generation can be run interactively; otherwise it exits. The idle guard unloads the model after the timeout, and reloads it transparently on the next `get_pipe()` call.


  Speed vs Quality Trade-off for Your Hardware

  Quality  │  cogvideox-2b 50 steps   ████████████████████  25–30 min
           │  cogvideox-2b 25 steps   ████████████████░░░░  13 min
           │  ltx-video 20 steps      ████████████░░░░░░░░  5 min
           │  ltx-video-distilled     ████████░░░░░░░░░░░░  1–2 min
  Speed    └─────────────────────────────────────────────▶