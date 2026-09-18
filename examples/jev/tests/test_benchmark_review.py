# SPDX-License-Identifier: Apache-2.0
"""Deterministic scheduling regressions; no GPU or speed thresholds required."""
import asyncio

import pytest

from helpers import request
from sglang_jev.benchmark import benchmark, summarize
from sglang_jev.contracts import ExperimentError


def fixture(name, *, instructions=None):
    data = request().model_dump()
    data["questions"] = {"fault": data["questions"]["fault"]}
    data["state"] = {"job": name}
    if instructions is not None:
        data["questions"]["fault"]["instructions"] = instructions
    return {"id": name, "request": data, "expected": {"fault": True}}


def response():
    return {
        "answers": {"fault": {
            "type": "noul", "status": "ok", "top_option": "yes",
            "probabilities": {"yes": 0.8, "no": 0.2}, "noul": 0.8,
            "max_probability": 0.8,
        }},
        "diagnostics": {"fallbacks": []},
    }


class Service:
    def __init__(self, action=None):
        self.action = action
        self.calls = []
        self.active = self.peak = self.starts = 0

    async def start(self):
        self.starts += 1

    async def read(self, req, *, cache_salt):
        name = req.state["job"]
        self.calls.append((name, cache_salt))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.action:
                await self.action(name)
            else:
                await asyncio.sleep(0)
            return response()
        finally:
            self.active -= 1


def test_fast_worker_replenishes_while_slow_request_is_pending():
    async def execute():
        third_started = asyncio.Event()
        slow_finished = asyncio.Event()

        async def action(name):
            if name == "slow":
                await third_started.wait()
                slow_finished.set()
            elif name == "third":
                assert not slow_finished.is_set()
                third_started.set()
            await asyncio.sleep(0)

        service = Service(action)
        fixtures = [fixture(k) for k in ("slow", "fast", "third", "fourth")]
        # A slice barrier deadlocks: 'third' cannot start until 'slow' finishes.
        records, report = await asyncio.wait_for(
            benchmark(service, fixtures, modes=["independent"], concurrency=2),
            timeout=1,
        )
        assert [r["fixture_id"] for r in records] == [f["id"] for f in fixtures]
        assert slow_finished.is_set()
        assert service.peak == 2 and service.active == 0
        assert report["scheduler"] == "bounded_worker_pool"
    asyncio.run(execute())


@pytest.mark.parametrize("concurrency", [1, 2, 20])
def test_every_job_runs_once_with_ordered_results_and_bounded_inflight(concurrency):
    async def execute():
        service = Service()
        fixtures = [fixture(str(i)) for i in range(7)]
        records, report = await benchmark(
            service, fixtures, modes=["packed"], repeats=2, concurrency=concurrency)
        assert [r["fixture_id"] for r in records] == [str(i) for i in range(7)] * 2
        assert len(service.calls) == 14
        assert 1 <= service.peak <= min(concurrency, 14)
        assert service.active == 0
        assert len({salt for _, salt in service.calls}) == 14
        phase = report["phases"][0]
        assert phase["attempts"] == phase["successes"] == 14
        assert phase["decisions_per_second"] == pytest.approx(
            phase["successful_decisions"] / phase["wall_seconds"])
    asyncio.run(execute())


def test_expected_failure_does_not_drop_queued_jobs_or_failure_denominators():
    async def execute():
        async def action(name):
            await asyncio.sleep(0)
            if name == "fail":
                raise ExperimentError("intentional_test_failure")
        service = Service(action)
        records, report = await benchmark(
            service, [fixture(k) for k in ("a", "fail", "c", "d")],
            modes=["packed"], concurrency=2)
        assert [r["status"] for r in records] == ["ok", "error", "ok", "ok"]
        phase = report["phases"][0]
        assert phase["attempts"] == 4 and phase["errors"] == 1
        assert phase["metrics_by_field"]["fault"]["coverage"] == 0.75
        assert phase["successful_decisions"] == 3
        assert service.active == 0 and len(service.calls) == 4
    asyncio.run(execute())


def test_unexpected_failure_cancels_and_drains_other_workers_before_returning():
    async def execute():
        slow_started = asyncio.Event()
        slow_cancelled = asyncio.Event()
        release = asyncio.Event()

        async def action(name):
            if name == "slow":
                slow_started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    slow_cancelled.set()
                    raise
            else:
                await slow_started.wait()
                raise RuntimeError("programming_error")
        service = Service(action)
        try:
            with pytest.raises(RuntimeError, match="programming_error"):
                await benchmark(service, [fixture("slow"), fixture("bug")],
                                modes=["packed"], concurrency=2)
            # Check before asyncio.run teardown, which would otherwise hide leaks.
            assert slow_cancelled.is_set()
            assert service.active == 0
        finally:
            release.set()
            await asyncio.sleep(0)
    asyncio.run(execute())


def test_caller_cancellation_drains_active_jobs_and_does_not_start_queued_jobs():
    async def execute():
        started = asyncio.Event()
        cancelled = []
        service = None
        async def action(name):
            if service.active == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(name)
        service = Service(action)
        task = asyncio.create_task(benchmark(
            service, [fixture(str(i)) for i in range(5)],
            modes=["packed"], concurrency=2))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(cancelled) == ["0", "1"]
        assert len(service.calls) == 2 and service.active == 0
    asyncio.run(execute())


@pytest.mark.parametrize("kwargs", [
    {"concurrency": True}, {"concurrency": 1.5}, {"repeats": True},
    {"variants": []}, {"warmup_repeats": -1},
])
def test_invalid_limits_fail_before_starting_service(kwargs):
    service = Service()
    with pytest.raises(ValueError):
        asyncio.run(benchmark(service, [fixture("a")], modes=["packed"], **kwargs))
    assert service.starts == 0 and service.calls == []


def test_shared_field_id_does_not_pool_different_question_definitions():
    async def execute():
        records, report = await benchmark(
            Service(), [fixture("a", instructions="Is a fault present?"),
                        fixture("b", instructions="Is routine maintenance due?")],
            modes=["independent", "packed"], concurrency=2)
        for phase in report["phases"]:
            metrics = phase["metrics_by_field"]
            assert len(metrics) == 2
            assert all(k.startswith("fault@") for k in metrics)
            assert all(v["attempted_n"] == 1 for v in metrics.values())
        assert all(r["question_hashes"]["fault"] for r in records)
        assert len(report["paired_comparisons"]) == 2
    asyncio.run(execute())


def test_policy_risk_coverage_does_not_reenable_abstentions_or_split_score_ties():
    records = []
    for i in range(4):
        r = {"mode": "packed", "status": "ok", "elapsed_ms": 1,
             "expected": {"fault": "yes" if i % 2 == 0 else "no"},
             "response": response()}
        records.append(r)
    records[2]["response"]["answers"]["fault"]["status"] = "abstained"
    records[2]["response"]["answers"]["fault"]["max_probability"] = 0.99
    records[3]["status"] = "error"
    curve = summarize(records)["packed"]["metrics_by_field"]["fault"]["risk_coverage"]
    assert len(curve) == 1
    assert curve[0]["coverage_of_attempted"] == 0.5
    assert curve[0]["risk"] == 0.5
    assert curve[0]["threshold"] == 0.8


def test_warmup_is_separate_from_measured_records_and_uses_same_cache_namespace():
    async def execute():
        service = Service()
        records, report = await benchmark(
            service, [fixture("a"), fixture("b")], modes=["packed"],
            repeats=2, concurrency=2, cache="warm", warmup_repeats=1)
        assert len(service.calls) == 6 and len(records) == 4
        assert len({salt for _, salt in service.calls}) == 1
        phase = report["phases"][0]
        assert phase["attempts"] == 4
        assert phase["warmup"]["attempts"] == 2
        assert phase["warmup"]["errors"] == 0
        assert phase["warmup"]["wall_seconds"] >= 0
    asyncio.run(execute())


def test_warmup_failures_are_reported_not_merged_with_measured_failures():
    async def execute():
        failed = False
        async def action(name):
            nonlocal failed
            if not failed:
                failed = True
                raise ExperimentError("warmup_failure")
        service = Service(action)
        records, report = await benchmark(
            service, [fixture("a")], modes=["packed"], cache="warm")
        phase = report["phases"][0]
        assert len(records) == 1 and records[0]["status"] == "ok"
        assert phase["errors"] == 0
        assert phase["warmup"]["errors"] == 1
        assert phase["warmup"]["failures"][0]["error"] == "warmup_failure"
    asyncio.run(execute())
