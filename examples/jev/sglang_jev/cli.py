# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from .json_io import load_json
from .benchmark import benchmark
from .compiler import Compiler, fingerprint_tokenizer
from .contracts import ReadRequest, canonical_json
from .runtime import IdentityExpectation, ReadService, SGLangEndpoint


def build_service(args) -> ReadService:
    from transformers import AutoTokenizer

    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be an immutable 40-hex model commit")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, revision=args.revision,
        local_files_only=not args.allow_download, use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    compiler = Compiler(tokenizer, tokenizer_hash=fingerprint_tokenizer(tokenizer),
                        max_input_tokens=args.max_input_tokens,
                        chat_template_kwargs=load_json(args.chat_template_kwargs))
    base = SGLangEndpoint(args.base_url, api_key=os.getenv("JEV_BASE_API_KEY"),
                          timeout=args.timeout, max_concurrency=args.concurrency)
    uno = (SGLangEndpoint(args.uno_url, api_key=os.getenv("JEV_UNO_API_KEY"),
                          timeout=args.timeout, max_concurrency=args.concurrency)
           if args.uno_url else None)
    return ReadService(compiler, base, uno=uno, cache_salt=args.cache_salt,
                       expectation=IdentityExpectation(args.model, args.revision, args.uno_adapter_path))


def main():
    parser = argparse.ArgumentParser(description="No-training SGLang typed-read experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="Exact --model-path reported by SGLang")
    common.add_argument("--revision", required=True)
    common.add_argument("--tokenizer", help="Defaults to --model; uses the same revision")
    common.add_argument("--base-url", default="http://127.0.0.1:30000")
    common.add_argument("--uno-url")
    common.add_argument("--uno-adapter-path", help="Pinned local adapter path reported by UNO")
    common.add_argument("--allow-download", action="store_true")
    common.add_argument("--trust-remote-code", action="store_true")
    common.add_argument("--chat-template-kwargs", default='{"enable_thinking":false}')
    common.add_argument("--max-input-tokens", type=int, default=32768)
    common.add_argument("--timeout", type=float, default=120.0)
    common.add_argument("--concurrency", type=int, default=4)
    common.add_argument("--cache-salt")
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path)
    serve = sub.add_parser("serve", parents=[common])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8011)
    bench = sub.add_parser("benchmark", parents=[common])
    bench.add_argument("--fixtures", type=Path, required=True)
    bench.add_argument("--modes", default="independent,packed,ar_vector,constrained_vector,uno_vector")
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--variants", default="original",
                       help="original,reverse_fields,shuffle_options,placeholder_underscore")
    bench.add_argument("--order-seed", type=int, default=20260917)
    bench.add_argument("--cache", choices=["cold", "warm"], default="cold")
    bench.add_argument("--warmup-repeats", type=int, default=1,
                       help="Warm-only unmeasured repeats per fixture and phase")
    bench.add_argument("--output", type=Path, required=True)
    sub.add_parser("schema")
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(ReadRequest.model_json_schema(), indent=2))
        return
    if args.command == "serve" and args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("JEV_API_KEY"):
        parser.error("non-loopback binding requires JEV_API_KEY; use TLS at a trusted proxy")
    if args.uno_url and not args.uno_adapter_path:
        parser.error("--uno-url requires --uno-adapter-path to check adapter identity")
    if getattr(args, "output", None) and args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.command == "benchmark" and args.output.with_suffix(args.output.suffix + ".summary.json").exists():
        parser.error("summary output already exists; choose a new path")
    service = build_service(args)
    if args.command == "serve":
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(service, api_key=os.getenv("JEV_API_KEY")),
                    host=args.host, port=args.port, access_log=False)
        return

    async def execute():
        try:
            if args.command == "run":
                response = await service.read(ReadRequest.model_validate(load_json(args.request.read_bytes())))
                text = json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False)
                if args.output:
                    # Refuse accidental overwrite of previous experimental evidence.
                    with args.output.open("x") as f:
                        f.write(text + "\n")
                else:
                    print(text)
            else:
                fixtures = [load_json(line) for line in args.fixtures.read_text().splitlines() if line.strip()]
                records, report = await benchmark(service, fixtures, modes=args.modes.split(","),
                                                   repeats=args.repeats, concurrency=args.concurrency,
                                                   cache=args.cache,
                                                   variants=args.variants.split(","),
                                                   order_seed=args.order_seed,
                                                   warmup_repeats=args.warmup_repeats)
                with args.output.open("x") as f:
                    for record in records:
                        f.write(canonical_json(record) + "\n")
                report_path = args.output.with_suffix(args.output.suffix + ".summary.json")
                with report_path.open("x") as f:
                    json.dump(report, f, indent=2, allow_nan=False)
                print(json.dumps(report, indent=2, allow_nan=False))
        finally:
            await service.close()
    asyncio.run(execute())
