# SPDX-License-Identifier: Apache-2.0
"""Strict, transport-independent contracts for the no-training experiment."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Mode = Literal[
    "independent", "packed", "ar_vector", "uno_vector", "constrained_vector"
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Question(StrictModel):
    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1, max_length=8192)
    criteria: dict[str, str | None] | list[str] | None = None
    # Explicit host extension. Jev's public questions are independent.
    depends_on: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_criteria(self):
        if not self.instructions.strip():
            raise ValueError("instructions must include the complete question")
        c = self.criteria
        if self.type == "noul":
            if c is not None and (not isinstance(c, dict) or set(c) != {"yes", "no"}):
                raise ValueError("noul is binary; criteria must contain exactly yes/no")
        elif self.type == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= 255:
                raise ValueError("choice requires a map of 2..255 options")
            if any(not k.strip() or len(k) > 256 for k in c):
                raise ValueError("choice names must be non-empty and <=256 characters")
        else:
            if not isinstance(c, list) or not 2 <= len(c) <= 10:
                raise ValueError("score requires 2..10 ordered level descriptions")
            if any(not s.strip() for s in c):
                raise ValueError("score descriptions cannot be empty")
        descriptions = c.values() if isinstance(c, dict) else c or []
        if any(v is not None and len(v) > 8192 for v in descriptions):
            raise ValueError("criterion descriptions must be <=8192 characters")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("duplicate dependencies")
        return self

    def options(self) -> tuple[tuple[str, str], ...]:
        if self.type == "noul":
            c = self.criteria or {}
            return (("yes", c.get("yes") or "The answer is yes."),
                    ("no", c.get("no") or "The answer is no."))
        if self.type == "choice":
            return tuple((k, v or k) for k, v in self.criteria.items())
        return tuple((str(i), text) for i, text in enumerate(self.criteria))


class ReadPolicy(StrictModel):
    mode: Mode = "independent"
    placeholder: Literal["?", "_", "~"] = "?"
    field_order: list[str] | None = None
    option_seed: int | None = None
    # A chosen policy threshold, not a learned calibration guarantee.
    min_allowed_mass: float = Field(default=0.0, ge=0.0, le=1.0)
    min_max_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    on_invalid_vector: Literal["error", "independent"] = "error"
    # Defaults stay greedy; do not call repeated greedy runs independent samples.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class Constraint(StrictModel):
    kind: Literal["nondecreasing", "forbid_top_options"]
    fields: list[str] = Field(min_length=2, max_length=64)
    options: list[str] | None = None
    tolerance: float = Field(default=0.0, ge=0.0, le=1.0)


class ReadRequest(StrictModel):
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    read_policy: ReadPolicy = Field(default_factory=ReadPolicy)
    constraints: list[Constraint] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_request(self):
        if len(canonical_json(self.state).encode()) > 1_000_000:
            raise ValueError("state exceeds the 1 MB experiment limit")
        ids = set(self.questions)
        if any(not k.strip() or len(k) > 128 for k in ids):
            raise ValueError("question IDs must be non-empty and <=128 characters")
        for name, q in self.questions.items():
            if name in q.depends_on or not set(q.depends_on) <= ids:
                raise ValueError("unknown or self-referential dependency")
        order = self.read_policy.field_order
        if order is not None and (len(order) != len(ids) or set(order) != ids):
            raise ValueError("field_order must be an exact permutation of question IDs")
        self.levels()  # Reject cycles before any inference.
        for c in self.constraints:
            if len(set(c.fields)) != len(c.fields) or not set(c.fields) <= ids:
                raise ValueError("constraint fields must be known and unique")
            if c.kind == "nondecreasing":
                if c.options is not None or any(self.questions[k].type == "choice" for k in c.fields):
                    raise ValueError("nondecreasing is only for numeric noul/score values")
            elif c.options is None or len(c.options) != len(c.fields):
                raise ValueError("forbid_top_options needs one option per field")
            elif any(v not in dict(self.questions[k].options())
                     for k, v in zip(c.fields, c.options)):
                raise ValueError("constraint references an undeclared option")
        return self

    def levels(self) -> list[list[str]]:
        remaining = list(self.read_policy.field_order or self.questions)
        done: set[str] = set()
        levels = []
        while remaining:
            level = [k for k in remaining if set(self.questions[k].depends_on) <= done]
            if not level:
                raise ValueError("dependency cycle")
            levels.append(level)
            done.update(level)
            remaining = [k for k in remaining if k not in done]
        return levels

    def schema_hash(self) -> str:
        # Preserve declared question/option order; canonical_json sorts map keys.
        return digest({"questions": [(k, q.model_dump(), q.options())
                                     for k, q in self.questions.items()],
                       "constraints": [c.model_dump() for c in self.constraints]})


class ExperimentError(RuntimeError):
    """Safe public exception: never include prompts or upstream response bodies."""


class ProtocolError(ExperimentError):
    pass


class InvalidVector(ExperimentError):
    pass

