# Jev-style typed reads on SGLang: no-training experiment

This self-contained example compares **independent slot reads, packed schema
slots, and compact answer vectors generated with native UNO and verified by its
base autoregressive pathway**. It uses existing SGLang endpoints. It does not
train an adapter, alter the scheduler, monkeypatch the sampler, or claim that a
causal model supports DiffusionGemma-style bidirectional inpainting.

The code targets the API at SGLang commit
`4f52a2756328df5c27c9c6b76011805f7f40e8f8`. It is an experimental client/API and
benchmark, not a clinically validated classifier or a drop-in implementation of
the proprietary Jev model. CPU tests check contracts using a fake transport;
real model quality, K2 UNO compatibility, GPU throughput and calibration require
the opt-in live tests and experiments below.

## What is implemented

| Mode | Execution | Meaning of returned probabilities |
| --- | --- | --- |
| `independent` | One base prefill-only request per question; batched concurrently | Each question sees the state and its own definition only |
| `packed` | One base prefill-only request per compatible dependency group | Causal conditionals given all definitions and earlier placeholders |
| `ar_vector` | Base generates a compact vector; base rescores the accepted token IDs | Base conditionals along the actual generated vector |
| `constrained_vector` | Same, with a regex on base generation only | Raw, unconstrained base probabilities from separate rescoring |
| `uno_vector` | Native UNO generates and internally AR-verifies a vector; base rescores it | Raw base conditionals, **not** draft probabilities |

For `uno_vector`:

```text
state + question definitions + codebook
                    |
             "Answer:\n"
                    |
       released UNO adapter drafts blocks
                    |
       native base-AR accept/reject/correct
                    |
           accepted vector: A B 2
                    |
        exact token/scaffold validation
                    |
       separate base-only prefill scoring
                    |
       typed distributions + diagnostics
```

Native UNO may require several draft/verify cycles. The client reports the
observed `spec_verify_ct`; it does not claim one parallel pass or fabricate raw
`q` logits. Separators may take tokens, so block width is **not** field count.
No synthetic `ROOT` token is added: the stock sampler owns its seed/root logic.

## Typed contract

The external shape follows Jev's `state` plus `questions` terminology. Opaque
question IDs do not enter independent model prompts.

* `noul`: probability of **yes**, with exactly two alternatives, yes/no. It is
  neither a Boolean output nor a three-class unknown detector.
* `choice`: a map of 2 to 255 semantic names to descriptions (or null). The
  compiler fails if this tokenizer cannot represent that many stable codes.
* `score`: 2 to 10 ordered descriptions, indexed **0 through K-1**. Return the
  distribution, expected index, top level and legend.

`depends_on`, `read_policy`, and `constraints` are explicit host extensions,
not claims about Jev's public interface. `decision` is null when the host abstains;
raw distributions remain visible. No `confidence` field is synthesized and no
calibration is applied. `allowed_mass` measures mass on the code tokens, which
also reflects formatting/token priors. It is not a validated measure of truth or
evidence sufficiency.

A minimal request:

```json
{
  "state": {"note": "Synthetic example: the pump is currently leaking."},
  "questions": {
    "current_fault": {
      "type": "noul",
      "instructions": "Is a current mechanical fault asserted? A denied, repaired, historical or suspected fault alone is negative."
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate the explicitly stated urgency.",
      "criteria": ["Routine", "Prompt", "Immediate"]
    }
  },
  "read_policy": {"mode": "uno_vector", "on_invalid_vector": "error"}
}
```

See `fixtures/request.json` and `fixtures/synthetic.jsonl`. Those four synthetic
records are plumbing examples, **not an evaluation dataset**. Add representative,
properly separated labeled records before drawing performance conclusions.

## Installation

Run the example separately from an already installed checkout of the target
SGLang revision. This package does not import SGLang or install GPU dependencies.

```bash
uv venv .venv-jev
uv pip install --python .venv-jev/bin/python -e 'examples/jev[cli,test]'
.venv-jev/bin/sglang-jev schema > /tmp/jev-request.schema.json
.venv-jev/bin/python -m pytest -q examples/jev/tests
```

The dependency ranges are compatibility ranges, not a hardware-qualified lock.
Record the installed environment with each experiment. Tokenizer/model downloads
are disabled by default in the client. Use `--allow-download` explicitly to permit
them; `--trust-remote-code` is also explicit, after reviewing the pinned code.

## Start base and UNO servers

Use immutable model and adapter revisions. The base-scoring server must have
speculative decoding **disabled**. The UNO server must report `UNO` and the same
base path, revision, dtype, quantization, and weight version. Do not hot-reload
weights during a run.

The following is an **unqualified launch recipe**, not a claim that K2's released
adapter has passed this fork's loader. Current SGLang UNO documentation names
Qwen3-8B as explicitly validated; K2 needs the live startup check.

```bash
export MODEL=IFM/K2-Horizon-7B
export REVISION='<immutable 40-hex base-model commit>'
export UNO_REVISION='<immutable 40-hex adapter commit>'

# Run using the same HF cache intended for the server, before offline execution.
export UNO_ADAPTER_PATH="$(python -c '
import os
from huggingface_hub import snapshot_download
print(snapshot_download("IFM/K2-Horizon-7B-Uno", revision=os.environ["UNO_REVISION"],
    allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
')"

# Terminal/job 1: reference scorer.
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL" --revision "$REVISION" --trust-remote-code \
  --dtype bfloat16 --tp-size 1 --attention-backend fa3 \
  --host 127.0.0.1 --port 30000

# Terminal/job 2: the released adapter, without any additional training.
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$MODEL" --revision "$REVISION" --trust-remote-code \
  --dtype bfloat16 --tp-size 1 --attention-backend fa3 \
  --speculative-algorithm UNO --uno-lora-path "$UNO_ADAPTER_PATH" \
  --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  --host 127.0.0.1 --port 30001
```

Use the cluster's scheduler and allocated devices, rather than bypassing resource
isolation. The example above assumes two visible GPUs only to avoid memory
contention between the two server processes. Sharing one GPU requires explicit
memory-budget qualification. **Two servers mean two resident model copies**;
report that cost. This is not the proposed single-weight-copy native fast path.

Check the adapter's own `adapter_config.json` and pinned base compatibility before
launch. Do not merge the conditional UNO adapter into the AR reference. If K2
loading fails, retain the exact error and environment as a compatibility failure;
do not silently switch checkpoints or models. The same client can separately test
a documented Qwen UNO pair, with its identity recorded as a different experiment.

## Run or serve

```bash
.venv-jev/bin/sglang-jev run \
  --model "$MODEL" --revision "$REVISION" --trust-remote-code \
  --base-url http://127.0.0.1:30000 \
  --uno-url http://127.0.0.1:30001 --uno-adapter-path "$UNO_ADAPTER_PATH" \
  --request examples/jev/fixtures/request.json --output /tmp/jev-result.json

.venv-jev/bin/sglang-jev serve \
  --model "$MODEL" --revision "$REVISION" --trust-remote-code \
  --base-url http://127.0.0.1:30000 \
  --uno-url http://127.0.0.1:30001 --uno-adapter-path "$UNO_ADAPTER_PATH" \
  --host 127.0.0.1 --port 8011

curl http://127.0.0.1:8011/v1/jev/reads \
  -H 'Content-Type: application/json' \
  --data-binary @examples/jev/fixtures/request.json
```

`fixtures/request.json` defaults to `independent`; edit `read_policy.mode` to select
`uno_vector`. The optional UNO endpoint is unnecessary for the other four modes.
Credentials come from `JEV_BASE_API_KEY`, `JEV_UNO_API_KEY`, and `JEV_API_KEY`.
A non-loopback API bind requires `JEV_API_KEY`; use TLS through an approved proxy.
The configured URLs must be direct SGLang origins, not `/v1` paths.

## Benchmark all arms and perturbations

```bash
.venv-jev/bin/sglang-jev benchmark \
  --model "$MODEL" --revision "$REVISION" --trust-remote-code \
  --base-url http://127.0.0.1:30000 \
  --uno-url http://127.0.0.1:30001 --uno-adapter-path "$UNO_ADAPTER_PATH" \
  --fixtures examples/jev/fixtures/synthetic.jsonl \
  --modes independent,packed,ar_vector,constrained_vector,uno_vector \
  --variants original,reverse_fields,shuffle_options,placeholder_underscore \
  --concurrency 4 --repeats 2 --cache cold \
  --output /tmp/jev-benchmark.jsonl
```

The JSONL records preserve failures, fallback flags and exact executed modes.
A sibling `.summary.json` reports sequential, seeded-randomized phase order,
wall-time decisions/s, p50/p95 latency, per-field accuracy/NLL/Brier, binary
AUROC/AP when both classes exist, ordinal MAE, descriptive risk/coverage, and
paired semantic-label L1/JS/top-option changes. Repeated runs are not independent
patients or subjects. No statistical confidence interval is asserted.

Cold cache means a fresh namespace per request, **not** a server restart or
weight-cache flush. Warm cache shares a namespace within a phase and includes
warm-up. Vector timing includes native generation, parsing and separate scoring.
Fallback time stays in its originating arm. Evaluate retained quality at equal
coverage, not tokens/s on a different task. Run multiple disjoint dataset seeds,
phase orders, output widths and concurrency levels before making speed claims.

## Correctness boundaries

* Full-template differential tokenization verifies one invariant position per
  option, not just `encode(prefix + code)` length. Unsupported capacity, shifted
  boundaries, duplicate IDs and token-budget overflow fail closed.
* For answer token position `s`, the base LM prediction originates at `s-1`.
  SGLang's returned prompt suffix begins with a null row. The client requests
  `min(slot_positions)-1` and checks **all** returned input token IDs and candidate
  IDs before indexing. Missing probabilities are errors, never filled with a
  guessed floor or renormalized top-k approximation.
* For candidate logprobs `l`, conditional probabilities use log-sum-exp; allowed
  mass is `sum(exp(l))` before restricting the support. Underflow retains
  `allowed_log_mass` and causes abstention, not fabricated probability evidence.
* UNO does not accept grammar or returned-logprob requests in the inspected
  implementation. Regex is used only on the base constrained comparator. The
  UNO endpoint receives neither. Base rescoring receives no grammar or logit bias.
* Invalid vectors are errors by default. Optional `on_invalid_vector=independent`
  reruns the **whole group** and labels that fallback. No punctuation repair,
  truncation, missing-field guess, or retry-until-valid behavior is hidden.
* Later vector probabilities condition on earlier **generated** answers, not on
  replacements selected after rescoring. The result includes both
  `generated_option` and `top_option`; it does not silently rewrite the prefix.
* Dependencies are evaluated in topological bands. Unresolved parents block their
  children. Declarative constraints cause abstention rather than rewriting raw
  probabilities; subsequent dependency invalidation is propagated.

## GPU readiness tests

```bash
export JEV_LIVE=1 JEV_MODEL="$MODEL" JEV_REVISION="$REVISION"
export JEV_BASE_URL=http://127.0.0.1:30000
export JEV_UNO_URL=http://127.0.0.1:30001
export JEV_UNO_ADAPTER_PATH="$UNO_ADAPTER_PATH"
export JEV_TRUST_REMOTE_CODE=1
.venv-jev/bin/python -m pytest -q examples/jev/tests/test_live.py
```

The tests do not download weights, start servers, or submit jobs. They require
operator-provided servers. The own-slot-token invariance test detects causal
logprob off-by-one errors; native UNO tests require a valid vector and observed
verification cycles without fallback. A failure is useful compatibility/format
information, not permission to conceal the failing arm.

## Privacy and limitations

No supplied clinical records or private infrastructure files are bundled. Request
payload logging is disabled in the example and `no_logs=true` is sent upstream;
also configure server/proxy logging appropriately. Receipts contain schema labels,
code mappings, hashes and runtime metadata, but not full state or prompt tokens.
Hashes **are not anonymization**, especially for predictable inputs. Results and
labels remain sensitive experiment artifacts.

The local tokenizer is fingerprinted and checked against a server tokenization
probe. That probe is an alignment smoke test, not cryptographic remote attestation.
Immutable model/tokenizer/adapter files and no hot reload remain deployment
preconditions. Cancellation attempts to abort only this client's request. The API
is loopback-first experimental infrastructure, not a hardened multi-tenant service.

See [design and scope](docs/design.md) for source mappings and deferred native
paths. No model training or clinical performance claim is included in this PR.
