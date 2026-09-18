# Design, source contract, and experiment acceptance

## Scope and revisions

This implementation follows the revised **no-additional-training** direction.
The earlier idea of a dedicated Jev LoRA is superseded for this experiment.
It preserves the requested compact-vector UNO arm rather than substituting base
slot scoring for it.

| Discussed component | Status in this example |
| --- | --- |
| Independent base single-slot probabilities | Implemented |
| Packed base schema blanks | Implemented, explicitly not independent marginals |
| Compact vector, released UNO drafts, base verifies | Implemented through existing native UNO generation |
| Raw base distributions for each accepted vector slot | Implemented through a second base-only server |
| Base vector and regex-constrained vector comparators | Implemented |
| Noul/Choice/Score, deterministic compiler, explicit host abstention | Implemented |
| Optional dependencies and declarative consistency checks | Implemented as host extensions |
| Permutation/placeholder experiments and timing/quality summaries | Implemented |
| Clinical task bank, calibration fit, external/site validation | Not supplied or claimed |
| Additional LoRA training, fixed-point refinement, MoE/head changes | Not required and not included |
| Raw draft-logit reads and q-versus-p diagnostics | Not exposed by this path |
| Single-copy native typed-read endpoint | Future optimization after qualification |
| Partly clamped/noised UNO canvas like DiffusionGemma | Not implemented; different model/inference contract |

The API is a separate experiment service under `examples/jev`, not a new endpoint
inside SGLang's scheduler. That keeps the runtime behavior identical for existing
users and provides a measurable reference before native integration.

## Primary sources inspected

1. [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), structured
   DiffusionGemma reads: seeded canvas, fixed-width answer codes, read-only
   denoising and exact candidate logprobs. This is inspiration, not copied code.
2. [GLiClass](https://github.com/knowledgator/gliclass): runtime label descriptions
   as model input. This experiment uses an existing LM head, not GLiClass weights
   or a new classifier head.
3. [Jev primitives](https://docs.typesafe.ai/primitives): `state`, independent
   `questions`, `noul`, `choice`, and zero-based `score`. Public types are preserved;
   proprietary model training/calibration are not recreated.
4. [K2-Horizon-7B-Uno model card](https://huggingface.co/IFM/K2-Horizon-7B-Uno):
   released conditional adapter used with its base model.
5. [UNO SGLang documentation](https://docs.sglang.io/docs/advanced_features/speculative_decoding#uno-decoding):
   deployment constraints and supported native sampling path.
6. [UNO reference implementation](https://github.com/ifm-ai/uno): clean seed,
   corrupted future block, conditional LoRA, base verification.

Implementation contracts at the inspected SGLang base commit:

* [UNO worker and its unsupported request checks](https://github.com/sgl-project/sglang/blob/4f52a2756328df5c27c9c6b76011805f7f40e8f8/python/sglang/srt/speculative/uno_worker_v2.py)
* [Native request fields](https://github.com/sgl-project/sglang/blob/4f52a2756328df5c27c9c6b76011805f7f40e8f8/python/sglang/srt/managers/io_struct.py)
* [Prompt-logprob result alignment](https://github.com/sgl-project/sglang/blob/4f52a2756328df5c27c9c6b76011805f7f40e8f8/python/sglang/srt/managers/scheduler_components/logprob_result_processor.py)
* [Server tokenization endpoint](https://github.com/sgl-project/sglang/blob/4f52a2756328df5c27c9c6b76011805f7f40e8f8/python/sglang/srt/entrypoints/openai/serving_tokenize.py)
* [Native UNO end-to-end tests](https://github.com/sgl-project/sglang/blob/4f52a2756328df5c27c9c6b76011805f7f40e8f8/test/registered/spec/uno/test_uno.py)

These support the serving primitives, not an assertion that K2 can reliably answer
arbitrary typed tasks without fine-tuning. That remains the experiment.

## Three different estimands

For state x, question j, and candidate answer a:

```text
Independent:
    p(a | x, question_j, its_codebook)

Packed:
    p(a | x, all_question_definitions, fixed_scaffold, earlier_placeholders)

Vector rescoring:
    p(a | x, all_question_definitions, generated_answers_before_j)
```

They are not interchangeable. Packed mode does not observe future outputs. A
causal model reading a literal question-mark token does not acquire a masked-LM
objective or bidirectional attention. Vector rescoring gives a conditional along
one sampled/greedy path, not marginalization over all earlier answers.

If the rescored top answer differs from a generated answer, later scores retain
the original generated conditioning. The client must not return a silently
rewritten joint vector and call it verified. It returns a typed distribution and
the generated option separately. Dependency bands are additional host calls and
include only declared parent decisions.

Native UNO's accept/reject procedure concerns the base AR **token distribution**.
It is not a clinical verifier, a correctness proof, or calibration. No extra
"stable after two rounds" correctness criterion is introduced.

## Candidate probabilities and numerical contract

For complete-vocabulary logprob l(a), and allowed code-token set A:

```text
log_mass = logsumexp(l(a), a in A)
conditional(a) = exp(l(a) - log_mass)
allowed_mass = exp(log_mass)
```

Missing candidates, NaN, positive invalid logprobs, duplicate IDs, and mass above
one beyond numerical tolerance are errors. No arbitrary floor fills missing
values. Finite underflow is retained in log space. Raw maximum probability,
entropy and allowed mass are diagnostics; they are not automatically calibrated
confidence or independent evidence of uncertainty.

Noul remains binary, and absence of documented evidence is not automatically
absence of the underlying condition unless the supplied instructions define it
that way. The host must preserve each task's labeling rule. A separate evidence
sufficiency question may be useful but is not inserted silently.

## Compatibility and test gates

### CPU gate

Unit and fake-transport integration tests cover schema validation, dependency
cycles, zero-based score expectations, long semantic names, token capacity,
full-template token invariance, exact row alignment, missing probabilities,
underflow, independent question-ID invariance, UNO routing, raw-logprob
separation, regex isolation, explicit fallback, identity mismatch, changed weights,
API authentication/redaction/body limits, and cancellation scope.

Passing these tests establishes client correctness against the inspected API
contract. It does not establish a working GPU kernel/model combination.

### First GPU gate

1. Pin and load the exact K2 base and released UNO adapter.
2. Confirm the local/server tokenizer probe and server identity metadata.
3. Run ordinary native UNO generation and preserve its statistics/errors.
4. Run independent and packed zero-completion-token scoring.
5. Verify that changing an input token at slot s does not change its score from
   s-1, within floating-point tolerance.
6. Run compact vectors with no fallback. Report parse failures and observed
   verification cycles, including zero-cycle requests.
7. Repeat with concurrent requests, changed lengths, cache salts and warmed
   prefixes to detect state leakage and alignment bugs.

### Scientific acceptance, not yet evaluated

Use a held-out task bank with diverse Noul, Choice and Score tasks; separate
training/development/calibration/test data even though this implementation does
not train weights. Predeclare acceptable quality loss, coverage and latency.
Keep code mappings, question order, state length and concurrency paired.

Judge speed at equal predictive quality and coverage, including the second model
copy, rescoring, tokenization, host validation and failed attempts. Report native
vector parse rate separately from model task accuracy. Do not substitute the four
synthetic fixtures for this study or treat repeated prompts as additional subjects.

## Deferred native integration

Only after the client reference works should a native path be considered:

```text
host schema compiler
         |
validated typed-read metadata
         |
shared base-model weights + existing conditional UNO adapter
         |
existing draft + base verification
         |
selected-position candidate gather on GPU
         |
compact typed result, request-local cleanup
```

Native integration would require explicit request metadata, scheduler completion
semantics, logprob export, row-alignment tests, cancellation/KV ownership tests,
and GPU differential tests against this reference. A seeded, partly clamped UNO
canvas needs its own experiment: replacing trained uniform-noise blocks by fixed
schema text is not automatically licensed by the DiffusionGemma PR. Keep fixed
text in the prompt and compare native continuation vectors first.
