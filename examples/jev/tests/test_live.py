# SPDX-License-Identifier: Apache-2.0
"""Opt-in readiness checks. They do not download weights or launch GPU jobs."""
import asyncio
import json
import os
from argparse import Namespace

import pytest

from sglang_jev.cli import build_service
from sglang_jev.contracts import ReadRequest

pytestmark = [pytest.mark.live, pytest.mark.skipif(
    os.getenv("JEV_LIVE") != "1", reason="requires explicitly configured live GPU servers")]


def live_service():
    required = ("JEV_MODEL", "JEV_REVISION")
    if any(not os.getenv(k) for k in required):
        pytest.fail("live tests require JEV_MODEL and JEV_REVISION")
    uno_url = os.getenv("JEV_UNO_URL")
    adapter = os.getenv("JEV_UNO_ADAPTER_PATH")
    if uno_url and not adapter:
        pytest.fail("JEV_UNO_URL requires JEV_UNO_ADAPTER_PATH")
    return build_service(Namespace(
        model=os.environ["JEV_MODEL"], revision=os.environ["JEV_REVISION"],
        tokenizer=os.getenv("JEV_TOKENIZER"), allow_download=False,
        trust_remote_code=os.getenv("JEV_TRUST_REMOTE_CODE") == "1",
        chat_template_kwargs=os.getenv("JEV_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":false}'),
        max_input_tokens=8192, base_url=os.getenv("JEV_BASE_URL", "http://127.0.0.1:30000"),
        uno_url=uno_url, uno_adapter_path=adapter, timeout=120.0,
        concurrency=2, cache_salt="jev-live-tests",
    ))


def fixture_request(mode):
    from pathlib import Path
    source = Path(__file__).parents[1] / "fixtures" / "request.json"
    data = json.loads(source.read_text())
    data["read_policy"]["mode"] = mode
    return ReadRequest.model_validate(data)


@pytest.mark.parametrize("mode", ["independent", "packed"])
def test_live_prefill_zero_tokens(mode):
    service = live_service()
    async def execute():
        try:
            result = await service.read(fixture_request(mode))
            assert all(u["completion_tokens"] == 0 for u in result["diagnostics"]["usage"])
            assert result["provenance"]["tokenizer_alignment_probe"] == "passed"
            assert len(result["answers"]) == 3
        finally:
            await service.close()
    asyncio.run(execute())


def test_live_own_slot_token_does_not_change_its_score():
    service = live_service()
    req = fixture_request("packed")
    async def execute():
        try:
            await service.start()
            plan = service.compiler.compile(req.state, list(req.questions.items()))
            slot = plan.slots[-1]
            changed = list(plan.input_ids)
            changed[slot.position] = slot.candidate_ids[0]
            a, _ = await service._score(plan, cache_salt="jev-live-invariance-a")
            b, _ = await service._score(plan, ids=tuple(changed), cache_salt="jev-live-invariance-b")
            # Causal score at s comes from s-1. A wrong offset fails this test.
            assert a[slot.field_id] == pytest.approx(b[slot.field_id], abs=0.03)
        finally:
            await service.close()
    asyncio.run(execute())


def test_live_uno_vector_without_fallback():
    if not os.getenv("JEV_UNO_URL"):
        pytest.skip("requires a live, released UNO adapter")
    service = live_service()
    async def execute():
        try:
            result = await service.read(fixture_request("uno_vector"))
            assert not result["diagnostics"]["fallbacks"]
            assert result["diagnostics"]["native_uno_verification"]["observed_cycles"] > 0
            assert all("generated_option" in a for a in result["answers"].values())
        finally:
            await service.close()
    asyncio.run(execute())
