# BF16 and Q8_0 subtitle benchmark

Measured on October 28, 2026, on an Apple M5 Pro Mac mini with 18 CPU cores and 64 GiB unified memory. The [JSON report](2026-10-28-qwen3-0.6b-m5-pro.json) retains every output, timing, token count, resource sample and model hash. The [Markdown report](2026-10-28-qwen3-0.6b-m5-pro.md) summarizes the measurements.

| Model variant | p50 workload latency | Cues/sec | Completion tokens/sec | Deterministic pass rate |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B BF16 | 0.960 s | 12.779 | 145.899 | 100% |
| Qwen3-0.6B Q8_0 | 0.584 s | 19.783 | 227.506 | 100% |

The workload is the 12 original English dialogue cues in [`benchmark.SOURCES`](../../src/subzero/benchmark.py), translated into Brazilian Portuguese. Each variant had one warmup and three measured repetitions, with four cues per translation request. Latency covers a complete workload, including structural retries; it excludes model loading, warmup, resource sampling overhead outside the translation timer, and judging. The runtime kept its prompt cache between requests. Three repetitions are a small baseline, not a stable tail-latency estimate.

Both variants used llama.cpp 0.8.0, build 11960, commit `8642541b2`, with identical server arguments: context 4096, one slot, eight CPU threads, seed 42, GPU layers 99, prediction limit 2048 and thinking disabled. A verbose startup check confirmed each file's precision, the Apple M5 Pro Metal device, and 29/29 layers offloaded to the GPU. The OpenAI-compatible client uses temperature 0.2. Q8_0 was quantized locally from the verified BF16 file, rather than sourced from unrelated weights.

The fixed BF16 judge scored all six captured outputs at 5/5 using temperature zero. This is a 0.6B model evaluating a short workload from its own model family, not human or independent-model validation. Initial prompt-only judge responses were invalid JSON. The report retains those outcomes as `initial_judge`; the same translation outputs were rescored with the strict JSON schema, leaving all translation measurements unchanged.

Resource samples are host-wide. A separate fixed BF16 judge remained resident alongside the selected translation server; background desktop use and reclaimable memory also affect the readings. Peak sampled host memory was 38.799 GiB for BF16 and 39.156 GiB for Q8_0; peak swap was 0.925 GiB for both. These values do not measure isolated model allocation or prove Q8_0 uses more memory. NVIDIA VRAM is unavailable on this Mac. Admission limits were 6 GiB minimum available memory and 4 GiB maximum swap usage. One shared compute lease covered the complete session; requests were sequential and owned servers were stopped afterward.

## Model provenance

Source: [`Mungert/Qwen3-0.6B-GGUF`](https://huggingface.co/Mungert/Qwen3-0.6B-GGUF/tree/905efeec5cb1f50bc98f912941ef26c8a3840225), pinned revision `905efeec5cb1f50bc98f912941ef26c8a3840225`.

| File | SHA-256 |
| --- | --- |
| `Qwen3-0.6B-bf16.gguf` | `9b80f0f40c2038f5e9a909ead575bad9f1c2f3839e2d8c82559e53084992ad23` |
| `Qwen3-0.6B-Q8_0.gguf` | `13f4b656ed17ac3548a620ec03a89c95fd0bf25d1ff10d784de1c0b99ac00245` |

## Reproduce

Install the base CLI and a compatible llama.cpp runtime. Download the pinned BF16 file, verify its SHA-256, then quantize with the recorded llama.cpp build:

```sh
curl -fL 'https://huggingface.co/Mungert/Qwen3-0.6B-GGUF/resolve/905efeec5cb1f50bc98f912941ef26c8a3840225/Qwen3-0.6B-bf16.gguf' -o Qwen3-0.6B-bf16.gguf
shasum -a 256 Qwen3-0.6B-bf16.gguf
llama-quantize Qwen3-0.6B-bf16.gguf Qwen3-0.6B-Q8_0.gguf Q8_0
```

Run a BF16 judge server on port 18082 and one translation server on port 18081, with the arguments below. Coordinate their entire lifetime with other local GPU jobs. The measured run used a shared `gpu-run` lease around both servers and the benchmark supervisor.

```sh
llama-server -m Qwen3-0.6B-bf16.gguf --port 18082 \
  --host 127.0.0.1 --alias subzero-benchmark --ctx-size 4096 \
  --parallel 1 --threads 8 --seed 42 --n-gpu-layers 99 --predict 2048 \
  --jinja --chat-template-kwargs '{"enable_thinking":false}'
```

Start the translation server with the same arguments and port 18081. Save this configuration as `bf16.json`, with `model_file` pointing to the downloaded file:

```json
{
  "repeats": 3,
  "warmup": 1,
  "batch_size": 4,
  "sample_interval": 0.25,
  "judge": {
    "provider": "openai",
    "model": "subzero-benchmark",
    "url": "http://127.0.0.1:18082/v1"
  },
  "variants": [{
    "provider": "openai",
    "model": "subzero-benchmark",
    "url": "http://127.0.0.1:18081/v1",
    "label": "Qwen3-0.6B BF16",
    "precision": "BF16",
    "runtime": "llama.cpp 0.8.0 build 11960 (8642541b2)",
    "model_file": "Qwen3-0.6B-bf16.gguf"
  }]
}
```

```sh
subzero benchmark --config bf16.json --output bf16-report.json
```

Stop the translation server, start it with `Qwen3-0.6B-Q8_0.gguf`, and run the same configuration with its label, precision and model file updated to Q8_0. Keep the BF16 judge unchanged. Each CLI invocation writes JSON and Markdown; the published baseline combines the two measured variant records. Stop both servers when finished.

The harness does not infer precision from a model alias or install runtime servers. `precision` and `runtime` are declared metadata; `model_file` adds the actual file digest. Tokens/sec comes from provider usage fields; omitted counts remain unavailable. The benchmark command itself requires no optional worker packages.
