# SPDX-License-Identifier: Apache-2.0
"""Paired experiments; errors and fallbacks remain in throughput denominators."""
from __future__ import annotations

import asyncio
import math
import random
import statistics
import time
import uuid
from collections import defaultdict

from .contracts import ExperimentError, ReadRequest
from .runtime import ReadService

VARIANTS = {"original", "reverse_fields", "shuffle_options", "placeholder_underscore"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = q * (len(values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def binary_ranking(pairs: list[tuple[float, bool]]) -> dict:
    """AUROC by tied ranks; AP at distinct score thresholds."""
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None,
                "reason": "requires_both_classes"}
    groups = defaultdict(list)
    for score, target in pairs:
        groups[score].append(target)
    rank_sum = 0.0
    before = 0
    for score in sorted(groups):
        labels = groups[score]
        rank_sum += sum(labels) * (before + (len(labels) + 1) / 2)
        before += len(labels)
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    tp = seen = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        added = sum(labels)
        tp += added
        seen += len(labels)
        ap += (added / positives) * (tp / seen)
    return {"auroc": auc, "average_precision": ap}


def _truth(value, question) -> str:
    if question.type == "noul" and isinstance(value, bool):
        return "yes" if value else "no"
    if question.type == "score" and type(value) is int:
        return str(value)
    if not isinstance(value, str):
        raise ValueError("fixture target must be a declared option, binary bool, or score index")
    return value


def make_request(fixture: dict, mode: str, variant: str) -> ReadRequest:
    source = fixture["request"]
    policy = {**source.get("read_policy", {}), "mode": mode}
    if variant == "reverse_fields":
        policy["field_order"] = list(reversed(source["questions"]))
    elif variant == "shuffle_options":
        policy["option_seed"] = 20260917
    elif variant == "placeholder_underscore":
        policy["placeholder"] = "_"
    elif variant != "original":
        raise ValueError("unknown benchmark variant")
    return ReadRequest.model_validate({**source, "read_policy": policy})


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["mode"]].append(record)
    summary = {}
    for mode, group in groups.items():
        successes = [r for r in group if r["status"] == "ok"]
        fields = defaultdict(list)
        for record in group:
            for key, truth in record.get("expected", {}).items():
                answer = (record["response"]["answers"][key]
                          if record["status"] == "ok" else None)
                fields[key].append((truth, answer))
        metrics = {}
        for key, items in fields.items():
            scored = [(truth, a) for truth, a in items if a and "probabilities" in a]
            selected = [(truth, a) for truth, a in scored if a["status"] == "ok"]
            out = {"attempted_n": len(items), "scored_n": len(scored),
                   "answered_n": len(selected), "coverage": len(selected) / len(items),
                   "selective_accuracy": (statistics.mean(a["top_option"] == y for y, a in selected)
                                            if selected else None)}
            if scored:
                kind = scored[0][1]["type"]
                out["type"] = kind
                out["accuracy_on_scored"] = statistics.mean(a["top_option"] == y for y, a in scored)
                out["nll"] = statistics.mean(-math.log(max(a["probabilities"][y], 1e-300)) for y, a in scored)
                out["nll_clipped_count"] = sum(a["probabilities"][y] < 1e-300 for y, a in scored)
                if kind == "noul":
                    out["brier"] = statistics.mean((a["noul"] - (y == "yes")) ** 2 for y, a in scored)
                    out.update(binary_ranking([(a["noul"], y == "yes") for y, a in scored]))
                else:
                    out["brier"] = statistics.mean(sum((v - (label == y)) ** 2
                                                       for label, v in a["probabilities"].items())
                                                    for y, a in scored)
                if kind == "score":
                    out["mae"] = statistics.mean(abs(a["score"] - int(y)) for y, a in scored)
                # Descriptive risk/coverage by raw max probability, not calibration.
                ordered = sorted(scored, key=lambda item: item[1]["max_probability"], reverse=True)
                out["risk_coverage"] = []
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    n = max(1, math.ceil(fraction * len(ordered)))
                    subset = ordered[:n]
                    out["risk_coverage"].append({"coverage_of_attempted": n / len(items),
                        "risk": statistics.mean(a["top_option"] != y for y, a in subset)})
            metrics[key] = out
        summary[mode] = {
            "attempts": len(group), "successes": len(successes), "errors": len(group) - len(successes),
            "fallback_requests": sum(bool(r["response"]["diagnostics"]["fallbacks"]) for r in successes),
            "latency_p50_ms": percentile([r["elapsed_ms"] for r in group], 0.5),
            "latency_p95_ms": percentile([r["elapsed_ms"] for r in group], 0.95),
            "metrics_by_field": metrics,
        }
    return summary


def paired_comparisons(records: list[dict]) -> dict:
    index = {(r["fixture_id"], r["repeat"], r["mode"], r["variant"]): r for r in records}
    pairs = defaultdict(list)
    for key, current in index.items():
        fixture, repeat, mode, variant = key
        if current["status"] != "ok":
            continue
        reference_key = ((fixture, repeat, "independent", "original") if variant == "original"
                         else (fixture, repeat, mode, "original"))
        reference = index.get(reference_key)
        if not reference or reference["status"] != "ok" or reference is current:
            continue
        label = (f"{mode}/original vs independent/original" if variant == "original"
                 else f"{mode}/{variant} vs {mode}/original")
        for field, a in current["response"]["answers"].items():
            b = reference["response"]["answers"][field]
            if "probabilities" not in a or "probabilities" not in b:
                continue
            p, q = a["probabilities"], b["probabilities"]
            if set(p) != set(q):
                raise ValueError("paired distributions have different semantic labels")
            js = 0.0
            for k in p:
                mid = (p[k] + q[k]) / 2
                if p[k] > 0:
                    js += 0.5 * p[k] * math.log(p[k] / mid)
                if q[k] > 0:
                    js += 0.5 * q[k] * math.log(q[k] / mid)
            pairs[(label, field)].append({"l1": sum(abs(p[k] - q[k]) for k in p),
                "js_nats": js, "top_flip": a["top_option"] != b["top_option"]})
    return {f"{label} :: {field}": {"paired_n": len(values),
        **{metric: statistics.mean(v[metric] for v in values) for metric in ("l1", "js_nats", "top_flip")}}
        for (label, field), values in pairs.items()}


async def benchmark(service: ReadService, fixtures: list[dict], *, modes: list[str],
                    repeats: int = 1, concurrency: int = 1, cache: str = "cold",
                    variants: list[str] | None = None,
                    order_seed: int = 20260917) -> tuple[list[dict], dict]:
    variants = variants or ["original"]
    if (not fixtures or not modes or repeats < 1 or concurrency < 1
            or cache not in {"cold", "warm"} or not set(variants) <= VARIANTS
            or len(set(modes)) != len(modes) or len(set(variants)) != len(variants)):
        raise ValueError("invalid benchmark limits/modes/variants")
    if len(fixtures) * repeats * len(modes) * len(variants) > 10_000:
        raise ValueError("benchmark is limited to 10,000 attempts per invocation")
    if len({f["id"] for f in fixtures}) != len(fixtures):
        raise ValueError("fixture IDs must be unique")
    normalized = []
    for f in fixtures:
        req = ReadRequest.model_validate(f["request"])
        expected = {}
        for field, target in f.get("expected", {}).items():
            if field not in req.questions:
                raise ValueError("fixture target references an unknown question")
            value = _truth(target, req.questions[field])
            if value not in dict(req.questions[field].options()):
                raise ValueError("fixture target is not a declared option")
            expected[field] = value
        normalized.append({**f, "expected": expected})
        for mode in modes:
            for variant in variants:
                make_request(f, mode, variant)  # validate before starting GPU work
    await service.start()
    namespace = "jev-bench-" + uuid.uuid4().hex
    phases = [(mode, variant) for mode in modes for variant in variants]
    random.Random(order_seed).shuffle(phases)
    records, reports = [], []
    started = time.perf_counter()

    async def one(fixture, mode, variant, repeat):
        req = make_request(fixture, mode, variant)
        salt = namespace + "-" + mode + "-" + variant
        if cache == "cold":
            salt += "-" + uuid.uuid4().hex
        t = time.perf_counter()
        row = {"fixture_id": fixture["id"], "repeat": repeat, "mode": mode,
               "variant": variant, "expected": fixture["expected"]}
        try:
            row.update(status="ok", response=await service.read(req, cache_salt=salt))
        except ExperimentError as e:
            row.update(status="error", error_type=type(e).__name__, error=str(e))
        row["elapsed_ms"] = (time.perf_counter() - t) * 1000
        return row

    for mode, variant in phases:
        phase_start = time.perf_counter()
        jobs = [(f, mode, variant, rep) for rep in range(repeats) for f in normalized]
        current = []
        for start in range(0, len(jobs), concurrency):
            current.extend(await asyncio.gather(*(one(*j) for j in jobs[start:start + concurrency])))
        elapsed = time.perf_counter() - phase_start
        report = summarize(current)[mode]
        decisions = sum(a["status"] == "ok" for r in current if r["status"] == "ok"
                        for a in r["response"]["answers"].values())
        report.update(mode=mode, variant=variant, wall_seconds=elapsed,
                      successful_decisions=decisions,
                      decisions_per_second=decisions / elapsed if elapsed else None,
                      attempts_per_second=len(current) / elapsed if elapsed else None)
        reports.append(report)
        records.extend(current)
    return records, {"phases": reports, "paired_comparisons": paired_comparisons(records),
                     "wall_seconds": time.perf_counter() - started, "concurrency": concurrency,
                     "cache": cache, "order_seed": order_seed,
                     "note": "Sequential randomized phases; wall-time throughput includes errors and fallback work. "
                     "Cold uses fresh request cache salts, not restarts. Warm includes warm-up. "
                     "Metrics are per field, not pooled cross-task AUROCs. NLL clipping is evaluation-only. "
                     "Repeated records are not independent subjects; no confidence intervals are asserted."}
