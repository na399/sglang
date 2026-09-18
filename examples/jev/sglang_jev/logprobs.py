# SPDX-License-Identifier: Apache-2.0
"""Read raw prompt logprobs without top-k approximations or invented floors."""
from __future__ import annotations

import math
from typing import Any

from .compiler import CompiledRead, Slot
from .contracts import ProtocolError, ReadPolicy, ReadRequest


def distribution(logprobs: list[float]) -> tuple[list[float], float, float]:
    if len(logprobs) < 2 or any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x > 1e-5 for x in logprobs
    ):
        raise ProtocolError("candidate logprobs must be finite and <=0")
    top = max(logprobs)
    log_mass = top + math.log(sum(math.exp(x - top) for x in logprobs))
    if log_mass > 1e-4:
        raise ProtocolError("candidate probability mass exceeds one")
    probabilities = [math.exp(x - log_mass) for x in logprobs]
    return probabilities, math.exp(min(0.0, log_mass)), min(0.0, log_mass)


def read_slot_logprobs(result: dict[str, Any], plan: CompiledRead,
                       input_ids: tuple[int, ...]) -> dict[str, list[float]]:
    meta = result.get("meta_info", {})
    expected = len(input_ids) - plan.start
    if meta.get("prompt_tokens") != len(input_ids):
        raise ProtocolError("server prompt length differs from compiled token IDs")
    rows = meta.get("input_token_ids_logprobs")
    actual_tokens = meta.get("input_token_logprobs")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ProtocolError("candidate logprob row count differs from requested suffix")
    if not isinstance(actual_tokens, list) or len(actual_tokens) != expected:
        raise ProtocolError("input token logprob row count differs from requested suffix")
    if (not actual_tokens or not isinstance(actual_tokens[0], list)
            or len(actual_tokens[0]) < 2 or actual_tokens[0][0] is not None):
        raise ProtocolError("expected SGLang's leading null input-logprob row")
    # Validate ALL returned token IDs, not merely the answer position.
    for offset, item in enumerate(actual_tokens):
        if not isinstance(item, list) or len(item) < 2 or item[1] != input_ids[plan.start + offset]:
            raise ProtocolError("input logprob token alignment mismatch")
    output = {}
    for slot in plan.slots:
        row = rows[slot.position - plan.start]
        if not isinstance(row, list):
            raise ProtocolError("answer slot has no candidate logprobs")
        values: dict[int, float] = {}
        for entry in row:
            if not isinstance(entry, list) or len(entry) < 2:
                raise ProtocolError("malformed candidate logprob entry")
            value, token_id = entry[:2]
            if type(token_id) is not int or token_id in values:
                raise ProtocolError("duplicate or malformed candidate token ID")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError("missing candidate probability; no imputation permitted")
            values[token_id] = float(value)
        if set(plan.candidate_ids) != set(values):
            raise ProtocolError("server did not return the exact requested candidate set")
        output[slot.field_id] = [values[t] for t in slot.candidate_ids]
    return output


def make_answer(slot: Slot, logprobs: list[float], policy: ReadPolicy, *,
                mode: str, conditioning: list[dict[str, str]] | None = None,
                generated_option: str | None = None) -> dict[str, Any]:
    probs, mass, log_mass = distribution(logprobs)
    p = dict(zip(slot.labels, probs))
    top_option = max(p, key=p.get)
    entropy = -sum(x * math.log(x) for x in probs if x > 0)
    sorted_p = sorted(probs, reverse=True)
    reasons = []
    if mass == 0 or mass < policy.min_allowed_mass:
        reasons.append("low_allowed_mass")
    if max(probs) < policy.min_max_probability:
        reasons.append("low_max_probability")
    answer = {
        "type": slot.question.type,
        "status": "abstained" if reasons else "ok",
        "abstention_reasons": reasons,
        "top_option": top_option,
        "probabilities": p,
        "allowed_mass": mass,
        "allowed_log_mass": log_mass,
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(probs)),
        "margin": sorted_p[0] - sorted_p[1],
        "max_probability": max(probs),
        "calibration": "not_applied",
        "executed_mode": mode,
        "conditioning": conditioning or [],
    }
    if slot.question.type == "noul":
        answer["noul"] = p["yes"]
        answer["value"] = p["yes"]
    elif slot.question.type == "choice":
        answer["choice"] = top_option
        answer["value"] = top_option
    else:
        score = sum(float(level) * probability for level, probability in p.items())
        answer["score"] = score
        answer["value"] = score
        answer["legend"] = {str(i): text for i, text in enumerate(slot.question.criteria)}
    if generated_option is not None:
        # Never substitute argmaxes into an already-scored causal prefix.
        answer["generated_option"] = generated_option
        answer["generated_option_probability"] = p[generated_option]
        answer["generated_matches_argmax"] = generated_option == top_option
    answer["decision"] = None if reasons else answer["value"]
    return answer


def apply_constraints(request: ReadRequest, answers: dict[str, dict]) -> list[dict]:
    violations = []
    # Evaluate against one eligibility snapshot. Mutating status inside this
    # loop would suppress later overlapping constraints and make order matter.
    eligible = {k for k, answer in answers.items() if answer["status"] == "ok"}
    invalid_fields = set()
    for c in request.constraints:
        if not set(c.fields) <= eligible:
            continue
        if c.kind == "nondecreasing":
            values = [answers[k]["value"] for k in c.fields]
            failed = any(a > b + c.tolerance for a, b in zip(values, values[1:]))
        else:
            failed = all(answers[k]["top_option"] == v for k, v in zip(c.fields, c.options))
        if failed:
            violations.append({"kind": c.kind, "fields": c.fields})
            invalid_fields.update(c.fields)
    for k in request.questions:
        if k in invalid_fields:
            answers[k]["status"] = "abstained"
            answers[k]["decision"] = None
            answers[k]["abstention_reasons"].append("constraint_violation")
    # Cross-level constraints can invalidate a parent after its child ran.
    for level in request.levels():
        for name in level:
            if any(answers[k]["status"] != "ok" for k in request.questions[name].depends_on):
                if answers[name]["status"] == "ok":
                    answers[name]["status"] = "abstained"
                    answers[name]["decision"] = None
                    answers[name]["abstention_reasons"].append("invalidated_dependency")
    return violations
