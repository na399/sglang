# SPDX-License-Identifier: Apache-2.0
import json
import math

import httpx

from sglang_jev.compiler import Compiler
from sglang_jev.contracts import ReadRequest
from sglang_jev.runtime import IdentityExpectation, ReadService, SGLangEndpoint


class CharTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"


class FakeSGLang:
    def __init__(self, uno=False):
        self.uno = uno
        self.calls = []
        self.generated = list(map(ord, "A B 2"))
        self.corrupt = None
        self.identity_override = {}
        self.weight_version = "frozen-test"

    def handler(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, body))
        if path == "/server_info":
            payload = {"model_path": "model", "revision": "a" * 40,
                       "dtype": "bfloat16", "quantization": None,
                       "version": "fake", "speculative_algorithm": "UNO" if self.uno else None,
                       "uno_lora_path": "/pinned/uno" if self.uno else None,
                       # Must never leak into provenance.
                       "launch_command": "--api-key SECRET"}
            payload.update(self.identity_override)
        elif path == "/model_info":
            payload = {"model_path": self.identity_override.get("model_path", "model"),
                       "weight_version": self.weight_version, "is_generation": True}
        elif path == "/v1/tokenize":
            payload = {"tokens": list(map(ord, body["prompt"]))}
            if self.corrupt == "tokenizer":
                payload["tokens"][0] += 1
        elif path == "/abort_request":
            return httpx.Response(200, json={})
        elif path == "/generate":
            assert body["no_logs"] is True
            if body["sampling_params"]["max_new_tokens"] > 0:
                if self.uno:
                    assert "regex" not in body["sampling_params"]
                    assert not body.get("return_logprob")
                payload = {"output_ids": self.generated,
                           "meta_info": {"prompt_tokens": len(body["input_ids"]),
                                         "completion_tokens": len(self.generated),
                                         "spec_verify_ct": 1 if self.uno else 0}}
            else:
                assert not self.uno, "prompt scoring must never reach stock UNO"
                ids = body["input_ids"]
                start = body["logprob_start_len"]
                candidates = body["token_ids_logprob"]
                rows = [None]
                actual = [[None, ids[start], None]]
                for pos in range(start + 1, len(ids)):
                    weights = [1 + (t % 13) + pos % 7 for t in candidates]
                    total = sum(weights)
                    lps = [math.log(0.8 * w / total) for w in weights]
                    rows.append([[lp, t, None] for lp, t in zip(lps, candidates)])
                    actual.append([dict(zip(candidates, lps)).get(ids[pos], -10), ids[pos], None])
                payload = {"output_ids": [], "meta_info": {
                    "prompt_tokens": len(ids), "completion_tokens": 0,
                    "input_token_logprobs": actual, "input_token_ids_logprobs": rows}}
                if self.corrupt == "missing":
                    for row in rows[1:]:
                        row.pop()
                elif self.corrupt == "shift":
                    actual[-1][1] += 1
                elif self.corrupt == "null":
                    rows[-1] = None
                elif self.corrupt == "nan":
                    for row in rows[1:]:
                        row[0][0] = None
        else:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=payload)


def make_service(base_fake=None, uno_fake=None, **compiler_kwargs):
    base_fake = base_fake or FakeSGLang()
    base = SGLangEndpoint("http://base", transport=httpx.MockTransport(base_fake.handler))
    uno = (SGLangEndpoint("http://uno", transport=httpx.MockTransport(uno_fake.handler))
           if uno_fake else None)
    compiler = Compiler(CharTokenizer(), tokenizer_hash="test-tokenizer", **compiler_kwargs)
    service = ReadService(compiler, base, uno=uno,
                          expectation=IdentityExpectation("model", "a" * 40, "/pinned/uno"))
    return service, base_fake


def request(mode="independent", **policy):
    return ReadRequest.model_validate({"state": {"note": "Synthetic mechanical-fault record."},
        "questions": {
            "fault": {"type": "noul", "instructions": "Is a current fault stated?"},
            "category": {"type": "choice", "instructions": "What category is stated?",
                         "criteria": {"mechanical": "A mechanical problem", "none": "No problem"}},
            "urgency": {"type": "score", "instructions": "Rate stated urgency.",
                        "criteria": ["Routine", "Prompt", "Immediate"]}},
        "read_policy": {"mode": mode, **policy}})
