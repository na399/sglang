# SPDX-License-Identifier: Apache-2.0
"""Exercise installed CLI behavior without loading models or opening sockets."""
import json
import sys

import pytest

from helpers import make_service, request
from sglang_jev import cli


def args(monkeypatch, command, *extra):
    monkeypatch.setattr(sys, "argv", [
        "sglang-jev", command, "--model", "model", "--revision", "a" * 40,
        *map(str, extra),
    ])


def test_schema_command_needs_no_model_or_transformers(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sglang-jev", "schema"])
    cli.main()
    schema = json.loads(capsys.readouterr().out)
    assert {"state", "questions"} <= set(schema["properties"])


def test_run_writes_typed_result_and_closes_clients(monkeypatch, tmp_path):
    source, output = tmp_path / "request.json", tmp_path / "result.json"
    source.write_text(json.dumps(request().model_dump()))
    service, _ = make_service()
    monkeypatch.setattr(cli, "build_service", lambda args: service)
    args(monkeypatch, "run", "--request", source, "--output", output)
    cli.main()
    result = json.loads(output.read_text())
    assert len(result["answers"]) == 3
    assert service.base.client.is_closed
    assert "Synthetic mechanical-fault record" not in output.read_text()


def test_benchmark_cli_writes_measured_records_and_separate_warmup(monkeypatch, tmp_path, capsys):
    source, output = tmp_path / "fixtures.jsonl", tmp_path / "results.jsonl"
    source.write_text(json.dumps({
        "id": "synthetic", "request": request().model_dump(),
        "expected": {"fault": True},
    }) + "\n")
    service, _ = make_service()
    monkeypatch.setattr(cli, "build_service", lambda args: service)
    args(monkeypatch, "benchmark", "--fixtures", source, "--output", output,
         "--modes", "independent,packed", "--cache", "warm", "--warmup-repeats", 2)
    cli.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(output.with_suffix(".jsonl.summary.json").read_text())
    assert len(records) == 2
    assert all(p["attempts"] == 1 and p["warmup"]["attempts"] == 2 for p in summary["phases"])
    assert json.loads(capsys.readouterr().out) == summary
    assert service.base.client.is_closed


@pytest.mark.parametrize("existing", ["records", "summary"])
def test_cli_does_not_overwrite_previous_evidence(monkeypatch, tmp_path, existing):
    output = tmp_path / "result.jsonl"
    protected = output if existing == "records" else output.with_suffix(".jsonl.summary.json")
    protected.write_text("retain this evidence")
    monkeypatch.setattr(cli, "build_service", lambda args: pytest.fail("must not construct service"))
    args(monkeypatch, "benchmark", "--fixtures", tmp_path / "unused.jsonl", "--output", output)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert protected.read_text() == "retain this evidence"


def test_nonloopback_api_without_key_is_rejected_before_loading_model(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(cli, "build_service", lambda args: pytest.fail("must not construct service"))
    args(monkeypatch, "serve", "--host", "0.0.0.0")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
