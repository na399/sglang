# SPDX-License-Identifier: Apache-2.0
"""Reject ambiguous or non-finite JSON before schema validation."""
import json


def load_json(text: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

