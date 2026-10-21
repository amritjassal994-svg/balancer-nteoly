import asyncio
import hmac
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from redis.asyncio import Redis
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
#
# BACKENDS: comma-separated base URLs of the real backend nodes, e.g.
#   BACKENDS=https://yourapp.justrunmy.app,https://yourapp.onrender.com,https://yourapp.fastapicloud.dev
# They are named node-1, node-2, ... in the order listed. Each backend has the
# same name set as NODE_ID in its env, so keep the order stable.
#
# REDIS_URL: same Redis the backends send their heartbeat to. Without it the
# balancer behaves exactly like before.
#
# PROXY_SHARED_SECRET: same value set on all backend nodes (auth/limiter.py).
# Stamped on every forwarded request so backends know X-Forwarded-For can be
# trusted. Also required (as the x-proxy-secret header) to see /_proxy/status.
# ---------------------------------------------------------------------------
BACKENDS = [b.strip().rstrip("/") for b in os.environ.get("BACKENDS", "").split(",") if b.strip()]
BACKEND_NAMES = {b: f"node-{i}" for i, b in enumerate(BACKENDS, start=1)}
PROXY_SHARED_SECRET = os.environ.get("PROXY_SHARED_SECRET", "")
REDIS_URL = os.environ.get("REDIS_URL", "")

if not BACKENDS:
    raise RuntimeError("Set BACKENDS to a comma-separated list of backend base URLs")

UNHEALTHY_COOLDOWN_SECONDS = 20  # how long we steer around a backend after it fails
MAX_ATTEMPTS = 2  # first try + one retry on a different backend
MAX_BUFFERED_BODY = 1_000_000  # bodies up to ~1 MB are buffered so they can be retried
RETRY_STATUSES = {502, 503, 504}  # gateway-type errors: node is down / waking up
IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS"}
REGISTRY_INTERVAL_SECONDS = 30
REGISTRY_STALE_SECONDS = 120

# read=120s because Notes/merging.py can take a while (fetches several PDFs
# from Telegram storage, then merges them) -- don't cut that off early.
PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

# Live state per backend, rebuilt on every process start (on purpose).
active_connections: dict[str, int] = {b: 0 for b in BACKENDS}
unhealthy_until: dict[str, float] = {b: 0.0 for b in BACKENDS}

alive: dict[str, dict] = {}
registry_updated_at = float("-inf")

redis_client = (
    Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
    )
    if REDIS_URL
    else None
)

# Keep-alive pool: reuses TLS connections to the backends, which is where most
# of the proxy's speed comes from. Short expiry avoids reusing connections that
# the hosting platform has already silently closed.
http_client = httpx.AsyncClient(
    timeout=PROXY_TIMEOUT,
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=20.0),
)


async def watch_registry():
    global alive, registry_updated_at
    while True:
        try:
            names = list(BACKEND_NAMES.values())
            values = await redis_client.mget([f"node:{name}" for name in names])
            alive = {name: json.loads(value) for name, value in zip(names, values) if value}
            registry_updated_at = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Registry read failed", exc_info=True)
        await asyncio.sleep(REGISTRY_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(watch_registry()) if redis_client else None
    yield
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if redis_client:
        await redis_client.aclose()
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

# Frontend talks to THIS app now, not the backend nodes directly -- so it
# needs the same open CORS policy the backend used to have.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://notely-edu.pages.dev"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hop-by-hop headers: meaningful only for one specific hop, never forwarded.
# (content-length is handled separately below.)
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def is_alive(backend: str) -> bool:
    if redis_client is None:
        return True
    if time.monotonic() - registry_updated_at > REGISTRY_STALE_SECONDS:
        return True
    return BACKEND_NAMES[backend] in alive


def pick_backend(exclude=()) -> str | None:
    """Least-connections, skipping backends that failed recently, backends
    that stopped sending their heartbeat, and any in `exclude` (already tried
    for this request). Ties are broken randomly so idle traffic is spread
    across nodes instead of always hitting the first one in BACKENDS. If every
    candidate is marked unhealthy we try anyway.
    """
    candidates = [b for b in BACKENDS if b not in exclude]
    if not candidates:
        return None
    now = time.monotonic()
    healthy = [b for b in candidates if unhealthy_until[b] <= now and is_alive(b)]
    pool = healthy or candidates
    return min(pool, key=lambda b: (active_connections[b], random.random()))


def mark_unhealthy(backend: str):
    unhealthy_until[backend] = time.monotonic() + UNHEALTHY_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------
def build_forward_headers(request: Request) -> dict:
    headers = {
        k.lower(): v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"
    }
    # Overwrite, never append -- a client could otherwise set their own
    # X-Forwarded-For to dodge the backend's rate limiter.
    headers["x-forwarded-for"] = request.client.host if request.client else "unknown"
    headers["x-proxy-secret"] = PROXY_SHARED_SECRET
    return headers


def target_url(backend: str, request: Request) -> str:
    """Forward the path and query exactly as the client sent them (no
    decode/re-encode round trip, so signed URLs and %2F etc. survive)."""
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    url = backend + raw_path.decode("latin-1")
    query = request.scope.get("query_string", b"")
    if query:
        url += "?" + query.decode("latin-1")
    return url


async def prepare_body(request: Request):
    """Returns (body, replayable).

    No body / small body -> buffered, so the request can be retried on another
    backend. Large or chunked uploads are streamed straight through and are
    never retried (the stream can only be read once).
    """
    length = request.headers.get("content-length")
    if "transfer-encoding" not in request.headers:
        if length is None or length == "0":
            return b"", True
        if length.isdigit() and int(length) <= MAX_BUFFERED_BODY:
            return await request.body(), True
    return None, False


def stream_back(upstream: httpx.Response, backend: str, method: str) -> StreamingResponse:
    released = False

    async def release():
        # Decrement first: if we're being cancelled, the awaits below may not
        # finish, but the connection counter must never leak.
        nonlocal released
        if released:
            return
        released = True
        active_connections[backend] -= 1
        await upstream.aclose()

    async def body_stream():
        try:
            # aiter_raw, NOT aiter_bytes: aiter_bytes would decompress gzip/br
            # while we still forward the content-encoding header, so the browser
            # would try to decompress plain bytes and fail.
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await release()

    response = StreamingResponse(
        body_stream(), status_code=upstream.status_code, background=BackgroundTask(release)
    )
    # Raw header list: keeps duplicate headers (e.g. several Set-Cookie) intact.
    response.raw_headers = [
        (k.lower(), v)
        for k, v in upstream.headers.raw
        if k.decode("latin-1").lower() not in HOP_BY_HOP
        and not (method == "HEAD" and k.lower() == b"content-length")
    ]
    return response


# ---------------------------------------------------------------------------
# Status endpoint -- registered BEFORE the catch-all routes below.
# Needs the x-proxy-secret header so it isn't public infra info.
#   curl -H "x-proxy-secret: <secret>" https://your-proxy/_proxy/status
# ---------------------------------------------------------------------------
@app.get("/_proxy/status")
async def proxy_status(request: Request):
    supplied = request.headers.get("x-proxy-secret", "").encode()
    if not PROXY_SHARED_SECRET or not hmac.compare_digest(supplied, PROXY_SHARED_SECRET.encode()):
        return Response(status_code=404)
    now = time.monotonic()
    return {
        b: {
            "active_connections": active_connections[b],
            "healthy": unhealthy_until[b] <= now,
            "alive": is_alive(b),
            "load": alive.get(BACKEND_NAMES[b], {}).get("load"),
        }
        for b in BACKENDS
    }


# ---------------------------------------------------------------------------
# HTTP proxy -- streams both ways so big PDF merges/downloads are never fully
# buffered in this proxy's small RAM. If a backend is down (can't connect) or
# answers 502/503/504, replayable requests are retried once on another node.
# ---------------------------------------------------------------------------
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_http(full_path: str, request: Request):
    method = request.method
    body, replayable = await prepare_body(request)
    attempts = MAX_ATTEMPTS if replayable else 1
    idempotent = method in IDEMPOTENT_METHODS
    tried: list[str] = []

    for attempt in range(attempts):
        backend = pick_backend(exclude=tried)
        if backend is None:
            break
        tried.append(backend)
        active_connections[backend] += 1
        last_try = attempt == attempts - 1 or len(tried) == len(BACKENDS)

        try:
            headers = build_forward_headers(request)
            if replayable:
                content = body
            else:
                content = request.stream()
                if "content-length" in request.headers:
                    headers["content-length"] = request.headers["content-length"]
            upstream_request = http_client.build_request(
                method, target_url(backend, request), headers=headers, content=content
            )
            upstream = await http_client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            active_connections[backend] -= 1
            mark_unhealthy(backend)
            logger.warning("Upstream request failed (%s): %r", backend, exc)
            # Safe to retry if it never reached the backend, or if repeating
            # the request can't do any harm (GET/HEAD/OPTIONS).
            never_reached = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
            if replayable and not last_try and (never_reached or idempotent):
                continue
            break
        except BaseException:
            # Client went away / task cancelled -- don't leak the counter.
            active_connections[backend] -= 1
            raise

        platform_404 = (
            upstream.status_code == 404
            and "json" not in upstream.headers.get("content-type", "")
        )
        if upstream.status_code in RETRY_STATUSES or platform_404:
            mark_unhealthy(backend)
            if replayable and (idempotent or platform_404) and not last_try:
                active_connections[backend] -= 1
                await upstream.aclose()
                continue

        return stream_back(upstream, backend, method)

    return Response(
        content=b'{"detail":"Upstream unavailable, try again"}',
        status_code=502,
        media_type="application/json",
    )