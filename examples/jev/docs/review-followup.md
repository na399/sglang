# PR #1 review follow-up and remaining acceptance work

## Changes in this revision

The review of `2950e10` identified a fixed-batch barrier in the load generator.
The replacement is a bounded worker pool, not one task per queued fixture.
Each worker immediately claims another job after completion; the result array
preserves original job order. A deterministic regression test blocks the first
request until the third starts with concurrency two. It fails with the old
barrier and passes with replenishment without relying on a speed threshold.

Additional correctness fixes:

| Area | Change |
| --- | --- |
| Failure cleanup | Cancel and await every active worker on an unexpected exception or caller cancellation. Expected experiment errors remain measured rows. |
| Warm-cache measurement | Explicit `--warmup-repeats`; warm-up attempts, failures and elapsed time are separate from measured phase statistics. |
| Metric identity | Preserve per-question definition hashes; do not pool unrelated questions sharing an opaque field ID. |
| Risk/coverage | Only policy-accepted outputs are eligible; tied thresholds are included as complete groups. Errors and abstentions remain in attempted denominators. |
| Constraints | Evaluate every constraint against one eligibility snapshot, then apply the union of invalidated fields and propagate dependency invalidation. |
| Fallback receipts | Global semantics and executed modes now reflect the actual path, not merely the requested vector mode. |
| Protocol failures | Reject malformed native response metadata and nonnumeric candidate logprobs as sanitized protocol errors. |
| HTTP lifecycle | Detect disconnects during body reading and capacity waits; queued disconnected clients do not launch inference. Non-ASCII credentials yield 401 rather than an exception. |
| Packaging and tests | Add CLI regression coverage, opt-in mixed-length/cache/concurrency parity checks, and a read-only-permissions CPU CI workflow. |

## Validation actually performed

Local interpreter: Python 3.13.5. CPU tests use the checked-in fake tokenizer and
HTTP transport. No model weights, tokenizer downloads, GPU endpoints or cluster
jobs were used.

```bash
PYTHONASYNCIODEBUG=1 python -W error::RuntimeWarning -m pytest -q examples/jev/tests
# 97 passed, 8 skipped

python -m compileall -q examples/jev/sglang_jev examples/jev/tests
PYTHONPATH=examples/jev python -m sglang_jev schema

# Uses already installed dependencies; not a clean registry dependency resolution.
python -m pip install --no-deps --no-build-isolation -e examples/jev
sglang-jev schema
```

The eight skipped tests require explicitly enabled live servers. The added GitHub
workflow targets Python 3.10 and 3.13, but adding a workflow is not evidence that
those CI jobs passed. Consult the PR checks for their actual status. Formatter
qualification and a clean hardware-specific dependency lock remain environment
setup work; no formatting-tool or GPU-stack qualification is claimed here.

## Remaining scope, explicitly not claimed complete

| Work | Current status / acceptance evidence needed |
| --- | --- |
| K2 base + released UNO adapter loading | Not run here. Pin both revisions and preserve launch/environment evidence. |
| Real tokenizer and prompt-logprob alignment | Live tests are present, but have not run. Compare own-slot invariance and cache/concurrency parity. |
| Five-arm throughput and task quality on H200 | Benchmark implementation is present. No throughput or model-quality result is claimed. Measure both resident model copies and separate rescoring. |
| Representative task bank and calibration | Synthetic plumbing fixtures only. Supply held-out task labels, separate calibration/test sets, and a declared risk/coverage target. |
| Raw UNO draft probabilities and q/p comparison | Not exposed by this client path. Requires native logit export and differential GPU tests. |
| Partly clamped/noised output-schema canvas | Still an unimplemented experimental path, not equivalent to packed causal prefill scoring. It needs explicit seed/scaffold/slot metadata and tests against the released adapter's actual alignment and attention contract. |
| Single-weight-copy native typed-read endpoint | Still not implemented. Requires scheduler completion semantics, candidate-logprob export, and request-local KV/cancellation tests. |
| Additional model training | Intentionally excluded; no new LoRA is required by the current reference experiment. |

The five implemented modes remain available without changing the existing SGLang
scheduler or sampler. They are a reference for the deferred native experiments,
not evidence that arbitrary clamped UNO inpainting already works. Do not enable a
new attention mask, skip native acceptance/rejection, or call draft probabilities
calibrated without explicitly labeling and validating that separate experiment.

The next executable gate is the opt-in live suite in `tests/test_live.py`, followed
by the paired benchmark on a representative held-out task bank. See
[design.md](design.md) for source mappings and the distinction between independent,
packed-placeholder and generated-vector conditional distributions.
