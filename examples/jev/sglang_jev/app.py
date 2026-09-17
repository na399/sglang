# SPDX-License-Identifier: Apache-2.0
"""Optional loopback-first API around the experiment service."""
from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .contracts import ExperimentError, InvalidVector, ReadRequest
from .runtime import ReadService
from .json_io import load_json



def create_app(service: ReadService, *, api_key: str | None = None,
               max_body_bytes: int = 2_000_000, max_requests: int = 4) -> FastAPI:
    if max_body_bytes < 1 or max_requests < 1:
        raise ValueError("limits must be positive")
    limit = asyncio.Semaphore(max_requests)

    @asynccontextmanager
    async def lifespan(_):
        try:
            await service.start()
            yield
        finally:
            await service.close()

    app = FastAPI(title="SGLang typed-read experiment", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "experimental": True,
                "uno_configured": service.uno is not None}

    @app.post("/v1/jev/reads")
    async def read(request: Request):
        if api_key is not None:
            header = request.headers.get("authorization", "")
            if not hmac.compare_digest(header, "Bearer " + api_key):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > max_body_bytes:
                return JSONResponse({"error": "request_too_large"}, status_code=413)
        try:
            parsed = ReadRequest.model_validate(load_json(bytes(data)))
        except (ValueError, ValidationError, RecursionError):
            # Pydantic's default error envelope can echo confidential inputs.
            return JSONResponse({"error": "invalid_typed_read_request"}, status_code=422)
        async with limit:
            task = asyncio.create_task(service.read(parsed))
            try:
                while not task.done():
                    if await request.is_disconnected():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        return JSONResponse({"error": "client_disconnected"}, status_code=499)
                    await asyncio.wait({task}, timeout=0.05)
                return await task
            except InvalidVector as e:
                return JSONResponse({"error": "invalid_vector", "reason": str(e)}, status_code=422)
            except ExperimentError as e:
                return JSONResponse({"error": "upstream_or_protocol_failure", "reason": str(e)}, status_code=502)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    return app
