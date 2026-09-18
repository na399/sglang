# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import FakeSGLang, make_service, request
from sglang_jev.app import create_app
from sglang_jev.contracts import ProtocolError, ReadRequest
from sglang_jev.logprobs import apply_constraints, distribution


def constraint_case(reverse=False):
    fields = {k: {"type": "noul", "instructions": f"Is {k} present?"}
              for k in ("a", "b", "c", "child")}
    fields["child"]["depends_on"] = ["c"]
    rules = [{"kind": "nondecreasing", "fields": ["a", "b"]},
             {"kind": "nondecreasing", "fields": ["b", "c"]}]
    req = ReadRequest.model_validate({"state": {}, "questions": fields,
                                      "constraints": rules[::-1] if reverse else rules})
    answers = {k: {"status": "ok", "value": p, "decision": p,
                   "abstention_reasons": [], "top_option": "yes"}
               for k, p in zip(fields, [0.9, 0.5, 0.1, 0.8])}
    return req, answers


def test_overlapping_constraints_all_evaluate_before_any_abstention():
    outputs = []
    for reverse in (False, True):
        req, answers = constraint_case(reverse)
        violations = apply_constraints(req, answers)
        assert len(violations) == 2
        assert all(a["decision"] is None for a in answers.values())
        assert answers["b"]["abstention_reasons"] == ["constraint_violation"]
        assert answers["child"]["abstention_reasons"] == ["invalidated_dependency"]
        outputs.append(answers)
    assert outputs[0] == outputs[1]


def test_constraints_do_not_evaluate_initially_blocked_or_abstained_fields():
    req, answers = constraint_case()
    answers["a"]["status"] = "blocked"
    violations = apply_constraints(req, answers)
    assert len(violations) == 1
    assert violations[0]["fields"] == ["b", "c"]
    assert answers["a"]["status"] == "blocked"


@pytest.mark.parametrize("bad", [None, True, "-1", math.nan, math.inf])
def test_bad_candidate_values_raise_safe_protocol_errors(bad):
    with pytest.raises(ProtocolError):
        distribution([bad, -2.0])


@pytest.mark.parametrize("bad_meta", [None, [], "not metadata"])
def test_malformed_metadata_is_a_protocol_error_not_an_unhandled_exception(bad_meta):
    async def execute():
        fake = FakeSGLang()
        original = fake.handler
        def handler(req):
            if req.url.path == "/generate":
                return httpx.Response(200, json={"meta_info": bad_meta})
            return original(req)
        fake.handler = handler
        service, _ = make_service(fake)
        try:
            with pytest.raises(ProtocolError):
                await service.read(request("packed"))
        finally:
            await service.close()
    asyncio.run(execute())


def test_fallback_global_semantics_match_executed_scoring():
    async def execute():
        uno = FakeSGLang(True)
        uno.generated = []
        service, _ = make_service(uno_fake=uno)
        try:
            result = await service.read(request("uno_vector", on_invalid_vector="independent"))
            assert result["mode"] == "uno_vector"  # preserve requested mode
            assert result["diagnostics"]["semantics"] == "independent_questions"
            assert result["diagnostics"]["executed_modes"] == ["independent"]
        finally:
            await service.close()
    asyncio.run(execute())


def test_non_ascii_auth_header_is_unauthorized_not_server_error():
    service, fake = make_service()
    with TestClient(create_app(service, api_key="test"), raise_server_exceptions=False) as client:
        result = client.post("/v1/jev/reads", json=request().model_dump(),
                             headers=[(b"Authorization", b"Bearer \xff")])
        assert result.status_code == 401
    assert not any(path == "/generate" for path, _ in fake.calls)


def test_disconnected_request_waiting_for_capacity_never_starts_inference():
    class Incoming:
        headers = {}
        def __init__(self, disconnected=False):
            self.disconnected = disconnected
        async def stream(self):
            yield json.dumps(request().model_dump()).encode()
        async def is_disconnected(self):
            return self.disconnected

    async def execute():
        entered, release = asyncio.Event(), asyncio.Event()
        class SlowService:
            uno = None
            calls = 0
            async def read(self, req):
                self.calls += 1
                entered.set()
                await release.wait()
                return {"done": True}
        service = SlowService()
        app = create_app(service, max_requests=1)
        handler = next(r.endpoint for r in app.routes if r.path == "/v1/jev/reads")
        first = asyncio.create_task(handler(Incoming()))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            result = await asyncio.wait_for(handler(Incoming(True)), 0.5)
            assert result.status_code == 499
            assert service.calls == 1
        finally:
            release.set()
            await first
    asyncio.run(execute())


def test_disconnect_arriving_during_capacity_wait_cancels_the_waiter():
    async def execute():
        entered, release, checked = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class SlowService:
            uno = None
            calls = 0
            async def read(self, req):
                self.calls += 1
                entered.set()
                await release.wait()
                return {"done": True}
        class Incoming:
            headers = {}
            disconnected = False
            def __init__(self, notify=False):
                self.notify = notify
            async def stream(self):
                yield json.dumps(request().model_dump()).encode()
            async def is_disconnected(self):
                if self.notify:
                    checked.set()
                return self.disconnected
        service = SlowService()
        app = create_app(service, max_requests=1)
        handler = next(r.endpoint for r in app.routes if r.path == "/v1/jev/reads")
        first = asyncio.create_task(handler(Incoming()))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            incoming = Incoming(notify=True)
            second = asyncio.create_task(handler(incoming))
            await asyncio.wait_for(checked.wait(), 0.5)
            incoming.disconnected = True
            result = await asyncio.wait_for(second, 0.5)
            assert result.status_code == 499
            assert service.calls == 1
        finally:
            release.set()
            await first
            if second is not None and not second.done():
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
    asyncio.run(execute())
