# SPDX-License-Identifier: Apache-2.0
"""HTTP experiment adapter over existing SGLang APIs; no sampler monkeypatching."""
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from . import __version__
from .compiler import CompiledRead, Compiler
from .contracts import ExperimentError, InvalidVector, ProtocolError, ReadRequest, digest
from .logprobs import apply_constraints, make_answer, read_slot_logprobs

# Never persist server_info wholesale: launch_command may contain credentials.
TOKENIZER_PROBE = "Answer:\n0: ?\n1: A B 2 C\nUnicode: α ไทย\n"


async def gather_scoped(*awaitables):
    tasks = [asyncio.create_task(a) for a in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


IDENTITY_FIELDS = ("model_path", "revision", "tokenizer_path", "tokenizer_revision",
                   "dtype", "quantization", "weight_version", "version",
                   "speculative_algorithm", "speculative_num_draft_tokens",
                   "speculative_num_steps", "speculative_eagle_topk", "uno_lora_path")


class SGLangEndpoint:
    def __init__(self, url: str, *, api_key: str | None = None,
                 timeout: float = 120.0, max_concurrency: int = 8,
                 transport: httpx.AsyncBaseTransport | None = None):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("use a direct SGLang origin, without /v1 or a URL path")
        if timeout <= 0 or max_concurrency < 1:
            raise ValueError("timeout and max_concurrency must be positive")
        self.url = url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=self.url, headers=headers,
                                       timeout=timeout, transport=transport,
                                       follow_redirects=False, trust_env=False)
        self.limit = asyncio.Semaphore(max_concurrency)
        self.identity: dict[str, Any] = {}

    async def close(self):
        await self.client.aclose()

    async def _json(self, path: str, payload: dict | None = None) -> Any:
        try:
            response = (await self.client.get(path) if payload is None
                        else await self.client.post(path, json=payload))
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as e:
            # Do not leak a response body, URL credentials, or an echoed state.
            status = getattr(getattr(e, "response", None), "status_code", None)
            raise ExperimentError(f"SGLang {path} failed (status={status})") from None
        if isinstance(value, dict) and "error" in value:
            raise ExperimentError(f"SGLang {path} returned an error envelope")
        return value

    async def inspect(self) -> dict[str, Any]:
        info, model = await gather_scoped(self._json("/server_info"), self._json("/model_info"))
        if not isinstance(info, dict) or not isinstance(model, dict):
            raise ProtocolError("invalid server identity response")
        merged = {**info, **model}  # live model identity takes precedence
        self.identity = {k: merged.get(k) for k in IDENTITY_FIELDS}
        if model.get("is_generation") is not True:
            raise ProtocolError("endpoint is not a generation model")
        return self.identity

    async def generate(self, payload: dict) -> dict:
        rid = "jev-" + uuid.uuid4().hex
        body = {**payload, "rid": rid, "stream": False, "no_logs": True,
                "return_text_in_logprobs": False}
        async with self.limit:
            try:
                result = await self._json("/generate", body)
            except (asyncio.CancelledError, ExperimentError):
                # Cancel only this request, never other users' work.
                with contextlib.suppress(Exception):
                    await asyncio.shield(asyncio.wait_for(
                        self.client.post("/abort_request", json={"rid": rid, "abort_all": False}), 3.0))
                raise
        if not isinstance(result, dict):
            raise ProtocolError("expected one native generation response")
        return result


@dataclass(frozen=True)
class IdentityExpectation:
    model_path: str
    revision: str
    uno_lora_path: str | None = None


class ReadService:
    def __init__(self, compiler: Compiler, base: SGLangEndpoint, *,
                 uno: SGLangEndpoint | None = None,
                 expectation: IdentityExpectation | None = None,
                 cache_salt: str | None = None):
        self.compiler, self.base, self.uno = compiler, base, uno
        self.expectation = expectation
        self.cache_salt = cache_salt or "jev-" + uuid.uuid4().hex
        self._ready = False
        self._identity_lock = asyncio.Lock()

    async def start(self):
        async with self._identity_lock:
            if self._ready:
                return
            base = await self.base.inspect()
            if base.get("speculative_algorithm") not in {None, "", "NONE"}:
                raise ProtocolError("base scoring endpoint must have speculative decoding disabled")
            if self.expectation:
                for key in ("model_path", "revision"):
                    if base.get(key) != getattr(self.expectation, key):
                        raise ProtocolError(f"base endpoint {key} differs from the pinned expectation")
            if self.uno:
                uno = await self.uno.inspect()
                if uno.get("speculative_algorithm") != "UNO":
                    raise ProtocolError("UNO endpoint must actually run native UNO")
                for key in ("model_path", "revision", "dtype", "quantization", "weight_version"):
                    if uno.get(key) != base.get(key):
                        raise ProtocolError(f"UNO and base endpoints differ in {key}")
                if self.expectation and self.expectation.uno_lora_path:
                    if uno.get("uno_lora_path") != self.expectation.uno_lora_path:
                        raise ProtocolError("UNO adapter path differs from pinned expectation")
            expected_tokens = list(self.compiler.encode(TOKENIZER_PROBE))
            for endpoint in (self.base, self.uno):
                if endpoint is None:
                    continue
                result = await endpoint._json("/v1/tokenize", {
                    "prompt": TOKENIZER_PROBE, "add_special_tokens": False})
                if not isinstance(result, dict) or result.get("tokens") != expected_tokens:
                    raise ProtocolError("local/server tokenizer alignment probe failed")
            self._ready = True

    async def check_live_identity(self):
        for endpoint in (self.base, self.uno):
            if endpoint is None:
                continue
            info = await endpoint._json("/model_info")
            if not isinstance(info, dict):
                raise ProtocolError("invalid live model identity response")
            if any(info.get(k) != endpoint.identity.get(k)
                   for k in ("model_path", "weight_version")):
                raise ProtocolError("model changed during the experiment; discard this read")

    async def close(self):
        await self.base.close()
        if self.uno:
            await self.uno.close()

    async def _score(self, plan: CompiledRead, *, ids: tuple[int, ...] | None = None,
                     cache_salt: str) -> tuple[dict[str, list[float]], dict]:
        ids = ids or plan.input_ids
        body = {"input_ids": list(ids), "cache_salt": cache_salt,
                "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
                "return_logprob": True, "logprob_start_len": plan.start,
                "top_logprobs_num": 0, "token_ids_logprob": list(plan.candidate_ids)}
        result = await self.base.generate(body)
        if result.get("meta_info", {}).get("completion_tokens") != 0:
            raise ProtocolError("scoring endpoint unexpectedly generated completion tokens")
        return read_slot_logprobs(result, plan, ids), self._usage(result)

    @staticmethod
    def _usage(result: dict) -> dict:
        meta = result.get("meta_info", {})
        return {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens",
                                    "spec_verify_ct", "spec_accept_length") if k in meta}

    async def read(self, request: ReadRequest, *, cache_salt: str | None = None) -> dict:
        await self.start()
        started = time.perf_counter()
        await self.check_live_identity()
        mode = request.read_policy.mode
        if mode == "uno_vector" and self.uno is None:
            raise ProtocolError("uno_vector requires a configured native UNO endpoint")
        salt = cache_salt or self.cache_salt
        answers, receipts, usage, fallbacks = {}, [], [], []
        generation_ms = score_ms = compile_ms = 0.0
        # Dependencies are materialized explicitly by the host, one level at a time.
        for level in request.levels():
            active = []
            for name in level:
                parents = request.questions[name].depends_on
                if any(answers[p]["status"] != "ok" for p in parents):
                    answers[name] = {"type": request.questions[name].type,
                                     "status": "blocked", "value": None, "decision": None,
                                     "abstention_reasons": ["unresolved_dependency"]}
                else:
                    active.append(name)
            if not active:
                continue
            # Different dependency sets must not become each other's hidden context.
            groups: dict[tuple[str, ...], list[str]] = {}
            for name in active:
                key = tuple(sorted(request.questions[name].depends_on))
                groups.setdefault(key, []).append(name)
            for parents, names in groups.items():
                dep = {k: answers[k]["decision"] for k in parents}
                kwargs = {"placeholder": request.read_policy.placeholder,
                          "option_seed": request.read_policy.option_seed,
                          "dependencies": dep}
                t = time.perf_counter()
                if mode == "independent":
                    plans = [self.compiler.compile(request.state, [(k, request.questions[k])], **kwargs)
                             for k in names]
                else:
                    plans = [self.compiler.compile(request.state, [(k, request.questions[k]) for k in names],
                                                   vector=mode.endswith("vector"), **kwargs)]
                compile_ms += (time.perf_counter() - t) * 1000
                generated = {}
                executed_mode = mode
                vector_ids = None
                if mode.endswith("vector"):
                    plan = plans[0]
                    params = {"max_new_tokens": plan.output_width,
                              "temperature": request.read_policy.temperature,
                              "top_p": 1.0, "top_k": -1}
                    if mode == "constrained_vector":
                        params["regex"] = plan.regex
                    endpoint = self.uno if mode == "uno_vector" else self.base
                    t = time.perf_counter()
                    result = await endpoint.generate({"input_ids": list(plan.prefix_ids),
                                                       "sampling_params": params,
                                                       "cache_salt": salt})
                    generation_ms += (time.perf_counter() - t) * 1000
                    usage.append({"phase": "generation", **self._usage(result)})
                    try:
                        ids = result.get("output_ids")
                        if not isinstance(ids, list):
                            raise InvalidVector("missing_output_ids")
                        generated = plan.validate_output(ids)
                        vector_ids = plan.prefix_ids + tuple(ids)
                    except InvalidVector as e:
                        if request.read_policy.on_invalid_vector == "error":
                            raise
                        fallbacks.append({"fields": names, "reason": str(e), "from": mode,
                                          "to": "independent", "attempt": plan.receipt()})
                        t = time.perf_counter()
                        plans = [self.compiler.compile(request.state, [(k, request.questions[k])], **kwargs)
                                 for k in names]
                        compile_ms += (time.perf_counter() - t) * 1000
                        executed_mode = "independent"
                        vector_ids, generated = None, {}
                t = time.perf_counter()
                scored = await gather_scoped(*[
                    self._score(plan, ids=vector_ids, cache_salt=salt) for plan in plans])
                score_ms += (time.perf_counter() - t) * 1000
                for plan, (logprobs, item_usage) in zip(plans, scored):
                    receipts.append(plan.receipt())
                    usage.append({"phase": "base_scoring", **item_usage})
                    previous = []
                    for slot in plan.slots:
                        answers[slot.field_id] = make_answer(
                            slot, logprobs[slot.field_id], request.read_policy,
                            mode=executed_mode, conditioning=list(previous),
                            generated_option=generated.get(slot.field_id))
                        if vector_ids is not None:
                            previous.append({"field_id": slot.field_id,
                                             "option": generated[slot.field_id]})
                        elif executed_mode == "packed":
                            previous.append({"field_id": slot.field_id,
                                             "placeholder": request.read_policy.placeholder})
        violations = apply_constraints(request, answers)
        await self.check_live_identity()
        native_cycles = sum(int(u.get("spec_verify_ct", 0)) for u in usage if u["phase"] == "generation")
        return {
            "protocol": "sglang-jev-read/0.1", "mode": mode,
            "answers": {k: answers[k] for k in request.questions},
            "diagnostics": {
                "semantics": ("independent_questions" if mode == "independent" else
                              "shared_questions_placeholder_prefix" if mode == "packed" else
                              "base_conditionals_along_generated_vector"),
                "native_uno_verification": {"configured": mode == "uno_vector",
                                            "observed_cycles": native_cycles if mode == "uno_vector" else 0},
                "draft_probabilities_exposed": False,
                "fallbacks": fallbacks, "constraint_violations": violations,
                "usage": usage,
                "timing_ms": {"compile": compile_ms, "generation": generation_ms,
                              "base_scoring": score_ms,
                              "total": (time.perf_counter() - started) * 1000},
            },
            "provenance": {
                "client_version": __version__, "schema_hash": request.schema_hash(),
                "state_hash": digest(request.state), "policy": request.read_policy.model_dump(),
                "base": self.base.identity, "uno": self.uno.identity if self.uno else None,
                "identity_check": "pinned" if self.expectation else "server_reported_only",
                "tokenizer_hash": self.compiler.tokenizer_hash,
                "tokenizer_alignment_probe": "passed",
                "cache_namespace_hash": digest(salt), "templates": receipts,
                "calibration": "not_applied",
            },
        }
