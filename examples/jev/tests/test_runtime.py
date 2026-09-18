# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import FakeSGLang, make_service, request
from sglang_jev.app import create_app
from sglang_jev.benchmark import benchmark, binary_ranking
from sglang_jev.contracts import ExperimentError, InvalidVector, ProtocolError, ReadRequest
from sglang_jev.runtime import SGLangEndpoint, gather_scoped


def run(service, req):
    async def execute():
        try:
            return await service.read(req)
        finally:
            await service.close()
    return asyncio.run(execute())


def generation_calls(fake):
    return [body for path, body in fake.calls if path == "/generate"]


@pytest.mark.parametrize("mode,count", [("independent", 3), ("packed", 1)])
def test_prefill_only_exact_rows(mode, count):
    service, fake = make_service()
    result = run(service, request(mode))
    calls = generation_calls(fake)
    assert len(calls) == count
    assert all(c["sampling_params"]["max_new_tokens"] == 0 for c in calls)
    for template in result["provenance"]["templates"]:
        assert template["logprob_start_len"] == min(s["position"] for s in template["slots"]) - 1
        call = next(c for c in calls if len(c["input_ids"]) == template["input_tokens"])
        for slot in template["slots"]:
            weights = [1 + t % 13 + slot["position"] % 7 for t in slot["candidate_ids"]]
            probabilities = list(result["answers"][slot["field_id"]]["probabilities"].values())
            assert probabilities == pytest.approx([w / sum(weights) for w in weights])
    serialized = json.dumps(result)
    assert "SECRET" not in serialized
    assert "Synthetic mechanical-fault record" not in serialized
    assert "confidence" not in serialized
    assert result["diagnostics"]["draft_probabilities_exposed"] is False


@pytest.mark.parametrize("mode", ["ar_vector", "constrained_vector", "uno_vector"])
def test_vector_native_generation_and_separate_base_scoring(mode):
    uno = FakeSGLang(uno=True) if mode == "uno_vector" else None
    service, base = make_service(uno_fake=uno)
    result = run(service, request(mode))
    generating = generation_calls(uno or base)[0]
    assert not generating.get("return_logprob")
    assert ("regex" in generating["sampling_params"]) == (mode == "constrained_vector")
    scoring = generation_calls(base)[-1]
    assert scoring["sampling_params"]["max_new_tokens"] == 0
    assert scoring["input_ids"] == generating["input_ids"] + (uno or base).generated
    assert result["answers"]["fault"]["generated_option"] == "yes"
    assert result["answers"]["category"]["generated_option"] == "none"
    assert result["answers"]["urgency"]["generated_option"] == "2"
    assert result["answers"]["category"]["conditioning"] == [{"field_id": "fault", "option": "yes"}]
    assert result["diagnostics"]["native_uno_verification"]["observed_cycles"] == (1 if uno else 0)
    assert "q" not in result["answers"]["fault"]


@pytest.mark.parametrize("bad", ["", "A B", "A B 2 extra", "Z B 2", "A,B,2"])
def test_invalid_vector_is_error_not_repaired(bad):
    uno = FakeSGLang(True)
    uno.generated = list(map(ord, bad))
    service, base = make_service(uno_fake=uno)
    with pytest.raises(InvalidVector):
        run(service, request("uno_vector"))
    assert not generation_calls(base)


def test_invalid_vector_fallback_is_explicit_whole_group():
    uno = FakeSGLang(True)
    uno.generated = []
    service, base = make_service(uno_fake=uno)
    result = run(service, request("uno_vector", on_invalid_vector="independent"))
    assert len(generation_calls(base)) == 3
    assert len(result["diagnostics"]["fallbacks"]) == 1
    assert all(a["executed_mode"] == "independent" for a in result["answers"].values())
    assert not any("generated_option" in a for a in result["answers"].values())


@pytest.mark.parametrize("key,value", [("revision", "b" * 40), ("model_path", "other"),
                                      ("dtype", "float16"), ("uno_lora_path", "/wrong")])
def test_identity_mismatch_is_rejected(key, value):
    uno = FakeSGLang(True)
    uno.identity_override[key] = value
    service, base = make_service(uno_fake=uno)
    with pytest.raises(ProtocolError):
        run(service, request("uno_vector"))
    assert not generation_calls(base)
    assert not generation_calls(uno)


@pytest.mark.parametrize("corruption", ["missing", "shift", "nan", "tokenizer"])
def test_bad_logprob_or_tokenizer_contract_rejected(corruption):
    fake = FakeSGLang()
    fake.corrupt = corruption
    service, _ = make_service(fake)
    with pytest.raises(ProtocolError):
        run(service, request())


def test_model_swap_during_read_discards_result():
    fake = FakeSGLang()
    original = fake.handler
    def handle(req):
        response = original(req)
        if req.url.path == "/generate":
            fake.weight_version = "changed"
        return response
    fake.handler = handle
    service, _ = make_service(fake)
    with pytest.raises(ProtocolError, match="model changed"):
        run(service, request("packed"))


def test_dependencies_block_without_calls_and_constraints_propagate():
    data = request().model_dump()
    data["questions"]["urgency"]["depends_on"] = ["fault"]
    data["read_policy"]["min_allowed_mass"] = 1.0
    service, fake = make_service()
    result = run(service, ReadRequest.model_validate(data))
    assert result["answers"]["urgency"]["status"] == "blocked"
    assert len(generation_calls(fake)) == 2
    data["read_policy"]["min_allowed_mass"] = 0.0
    service, _ = make_service()
    baseline = run(service, ReadRequest.model_validate(data))
    data["constraints"] = [{"kind": "forbid_top_options", "fields": ["fault", "category"],
        "options": [baseline["answers"][k]["top_option"] for k in ["fault", "category"]]}]
    service, _ = make_service()
    result = run(service, ReadRequest.model_validate(data))
    assert result["answers"]["fault"]["decision"] is None
    assert "invalidated_dependency" in result["answers"]["urgency"]["abstention_reasons"]


def test_failures_abort_only_own_request_without_leaking_body():
    calls = []
    def handle(req):
        calls.append((req.url.path, json.loads(req.content)))
        if req.url.path == "/generate":
            return httpx.Response(500, text="SECRET PATIENT STATE")
        return httpx.Response(200, json={})
    endpoint = SGLangEndpoint("http://base", transport=httpx.MockTransport(handle))
    async def execute():
        try:
            with pytest.raises(ExperimentError) as error:
                await endpoint.generate({"input_ids": [1]})
            assert "SECRET" not in str(error.value)
        finally:
            await endpoint.close()
    asyncio.run(execute())
    assert calls[-1][0] == "/abort_request"
    assert calls[-1][1] == {"rid": calls[0][1]["rid"], "abort_all": False}


def test_scoped_gather_cancels_siblings():
    cancelled = []
    async def slow():
        try:
            await asyncio.sleep(100)
        finally:
            cancelled.append(True)
    async def fail():
        await asyncio.sleep(0)
        raise ExperimentError("failure")
    with pytest.raises(ExperimentError):
        asyncio.run(gather_scoped(slow(), fail()))
    assert cancelled == [True]


def test_api_auth_validation_and_size_limits():
    service, _ = make_service()
    with TestClient(create_app(service, api_key="test", max_body_bytes=2000)) as client:
        path = "/v1/jev/reads"
        assert client.post(path, json=request().model_dump()).status_code == 401
        headers = {"Authorization": "Bearer test"}
        assert client.post(path, json=request().model_dump(), headers=headers).status_code == 200
        response = client.post(path, content='{"state":"SECRET", "state":3}', headers=headers)
        assert response.status_code == 422
        assert "SECRET" not in response.text
        assert client.post(path, content="X" * 2001, headers=headers).status_code == 413


def test_benchmark_records_failures_and_paired_metrics():
    service, fake = make_service(uno_fake=FakeSGLang(True))
    fixtures = [{"id": "s1", "request": request().model_dump(),
                 "expected": {"fault": True, "category": "none", "urgency": 2}},
                {"id": "s2", "request": request().model_dump(), "expected": {"fault": False}}]
    async def execute():
        try:
            return await benchmark(service, fixtures, modes=["independent", "packed", "uno_vector"],
                                   variants=["original", "reverse_fields"], concurrency=2)
        finally:
            await service.close()
    records, report = asyncio.run(execute())
    assert len(records) == 12
    assert len(report["phases"]) == 6
    assert sum(p["attempts"] for p in report["phases"]) == 12
    assert report["paired_comparisons"]
    for phase in report["phases"]:
        assert phase["decisions_per_second"] == pytest.approx(
            phase["successful_decisions"] / phase["wall_seconds"])
        assert phase["metrics_by_field"]["fault"]["attempted_n"] == 2
    serialized = json.dumps(report, allow_nan=False)
    assert "SECRET" not in serialized


@pytest.mark.parametrize("pairs,auc,ap", [
    ([(0.9, True), (0.1, False)], 1.0, 1.0),
    ([(0.1, True), (0.9, False)], 0.0, 0.5),
    ([(0.5, True), (0.5, False)], 0.5, 0.5),
])
def test_tied_binary_ranking(pairs, auc, ap):
    result = binary_ranking(pairs)
    assert result["auroc"] == pytest.approx(auc)
    assert result["average_precision"] == pytest.approx(ap)
