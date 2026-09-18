# SPDX-License-Identifier: Apache-2.0
import math

import pytest
from pydantic import ValidationError

from sglang_jev.app import load_json
from sglang_jev.compiler import Compiler
from sglang_jev.contracts import InvalidVector, ProtocolError, Question, ReadRequest
from sglang_jev.logprobs import distribution, make_answer
from helpers import CharTokenizer, request


@pytest.mark.parametrize("mode", ["independent", "packed", "ar_vector", "uno_vector", "constrained_vector"])
def test_request_modes(mode):
    assert request(mode).read_policy.mode == mode


@pytest.mark.parametrize("bad", [
    {"type": "noul", "instructions": "x", "criteria": {"yes": None, "no": None, "unknown": None}},
    {"type": "score", "instructions": "x", "criteria": ["only"]},
    {"type": "score", "instructions": "x", "criteria": ["ok"] * 11},
    {"type": "choice", "instructions": "x", "criteria": {"a": None}},
    {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(256)}},
    {"type": "noul", "instructions": "  "},
    {"type": "noul", "instructions": "x", "unexpected": True},
])
def test_invalid_question(bad):
    with pytest.raises(ValidationError):
        Question.model_validate(bad)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_rejects_ambiguity(text):
    with pytest.raises(ValueError):
        load_json(text)


def test_request_cycles_order_and_unknown_dependencies():
    data = request().model_dump()
    data["questions"]["fault"]["depends_on"] = ["category"]
    data["questions"]["category"]["depends_on"] = ["fault"]
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(data)
    data = request().model_dump()
    data["read_policy"]["field_order"] = ["fault", "fault", "urgency"]
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(data)
    data = request().model_dump()
    data["questions"]["fault"]["depends_on"] = ["missing"]
    with pytest.raises(ValidationError):
        ReadRequest.model_validate(data)


def test_schema_hash_preserves_option_order():
    a = request()
    data = a.model_dump()
    data["questions"]["category"]["criteria"] = {"none": "No problem", "mechanical": "A mechanical problem"}
    assert ReadRequest.model_validate(data).schema_hash() != a.schema_hash()


@pytest.mark.parametrize("vector", [False, True])
def test_complete_template_and_all_alternatives(vector):
    req = request()
    compiler = Compiler(CharTokenizer(), tokenizer_hash="x")
    plan = compiler.compile(req.state, list(req.questions.items()), vector=vector)
    assert plan.start == plan.slots[0].position - 1
    assert plan.start >= 0
    for slot in plan.slots:
        for token in slot.candidate_ids:
            changed = list(plan.input_ids)
            changed[slot.position] = token
            assert len(changed) == len(plan.input_ids)
    if vector:
        expected = list(map(ord, "A B 2"))
        assert plan.output_width == len(expected)
        assert plan.validate_output(expected) == {"fault": "yes", "category": "none", "urgency": "2"}
        with pytest.raises(InvalidVector):
            plan.validate_output(list(map(ord, "A\nB 2")))
        with pytest.raises(InvalidVector):
            plan.validate_output(list(map(ord, "A Z 2")))
        with pytest.raises(InvalidVector):
            plan.validate_output(expected + [ord("\n")])


def test_independent_prompt_ignores_question_id_and_siblings():
    req = request()
    compiler = Compiler(CharTokenizer(), tokenizer_hash="x")
    a = compiler.compile(req.state, [("private_id", req.questions["fault"])])
    b = compiler.compile(req.state, [("other_id", req.questions["fault"])])
    assert a.input_ids == b.input_ids
    assert a.template_hash == b.template_hash


def test_option_permutation_does_not_change_label_set():
    req = request()
    compiler = Compiler(CharTokenizer(), tokenizer_hash="x")
    a = compiler.compile(req.state, list(req.questions.items()), option_seed=1)
    b = compiler.compile(req.state, list(req.questions.items()), option_seed=7)
    assert [set(s.labels) for s in a.slots] == [set(s.labels) for s in b.slots]
    assert a.input_ids != b.input_ids


def test_long_names_do_not_need_single_token():
    q = Question(type="choice", instructions="Classify.", criteria={"a long semantic label": None, "another long label": None})
    plan = Compiler(CharTokenizer(), tokenizer_hash="x").compile({}, [("x", q)])
    assert len(plan.slots[0].candidate_ids) == 2


def test_fail_closed_when_tokenizer_capacity_insufficient():
    q = Question(type="choice", instructions="Classify.", criteria={str(i): None for i in range(255)})
    with pytest.raises(ProtocolError, match="stable one-token"):
        Compiler(CharTokenizer(), tokenizer_hash="x").compile({}, [("x", q)])


def test_complete_template_boundary_not_just_length():
    class MergeTokenizer(CharTokenizer):
        def encode(self, text, **kwargs):
            # Adversarial BPE-like change only after chat rendering, not in probe.
            ids = super().encode(text, **kwargs)
            if "<assistant>" in text and "0: B" in text:
                ids[-2] = ord("B")
                ids[-3] = 999
            return ids
    with pytest.raises(ProtocolError, match="invariant"):
        Compiler(MergeTokenizer(), tokenizer_hash="x").compile({}, [("x", request().questions["fault"])])


def test_token_budget_not_truncation():
    with pytest.raises(ProtocolError, match="not truncated"):
        Compiler(CharTokenizer(), tokenizer_hash="x", max_input_tokens=20).compile({}, [("x", request().questions["fault"])])


def test_underflow_does_not_destroy_conditional_probability():
    p, mass, logmass = distribution([-1001, -1000])
    assert sum(p) == pytest.approx(1)
    assert p[1] > p[0]
    assert mass == 0
    assert logmass < -999


@pytest.mark.parametrize("lp", [[math.nan, -1], [math.inf, -1], [None, -1], [0.1, -1], [0, 0]])
def test_invalid_logprobs(lp):
    with pytest.raises((ProtocolError, TypeError)):
        distribution(lp)


def test_noul_and_score_are_probabilities_and_expectations():
    req = request()
    plan = Compiler(CharTokenizer(), tokenizer_hash="x").compile(req.state, list(req.questions.items()))
    a = make_answer(plan.slots[0], [math.log(.72), math.log(.08)], req.read_policy, mode="independent")
    assert a["noul"] == pytest.approx(.9)
    assert a["allowed_mass"] == pytest.approx(.8)
    assert "confidence" not in a
    s = make_answer(plan.slots[2], [math.log(.2), math.log(.1), math.log(.5)], req.read_policy, mode="independent")
    assert s["score"] == pytest.approx(1.375)
    assert s["legend"] == {"0": "Routine", "1": "Prompt", "2": "Immediate"}


def test_abstention_not_an_extra_class():
    req = request(min_allowed_mass=.95)
    plan = Compiler(CharTokenizer(), tokenizer_hash="x").compile(req.state, [("fault", req.questions["fault"])])
    a = make_answer(plan.slots[0], [math.log(.7), math.log(.1)], req.read_policy, mode="independent")
    assert a["status"] == "abstained"
    assert a["decision"] is None
    assert set(a["probabilities"]) == {"yes", "no"}
