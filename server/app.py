"""MCP Image to 3D as a network service.

One process, one port:
  GET  /                  object list
  GET  /viewer/{job_id}   in-browser 3D Gaussian splat viewer
  GET  /objects/{file}    the .ply, the preprocessed input image and its list thumbnail
  POST /jobs              start a generation
  GET  /jobs/{job_id}     status + links
  GET  /health            readiness, GPU, limits
  /mcp                    MCP server (Streamable HTTP)

Generation takes tens of seconds on a consumer GPU, so it is a background
job, never a synchronous request. The pipeline is not thread-safe (it seeds globally and
mutates module state), so every call is serialised behind one lock.

The model leaves the GPU after IDOL_IDLE_UNLOAD_SECONDS without a job, so an idle service does
not hold VRAM other work could use; the next job reloads it. Only jobs count as activity: status
polls and gallery browsing never touch the model, so they must not keep it resident.
"""
from __future__ import annotations

import ctypes
import gc
import io
import json
import logging
import os
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mcp-image-to-3d")

# ---------------------------------------------------------------- settings ---

DATA = Path(os.environ.get("IDOL_DATA", "/data"))
OBJECTS = DATA / "objects"
TRASH = DATA / "trash"  # soft-deleted objects, restorable from the gallery
UPLOADS = DATA / "uploads"  # images the gallery page uploads to generate from
MODEL_DIR = os.environ.get("IDOL_MODEL_DIR", "/data/idol-local")
PORT = int(os.environ.get("IDOL_PORT", "8043"))
PUBLIC_BASE = os.environ.get("IDOL_PUBLIC_BASE_URL", "").rstrip("/")
MAX_GAUSSIANS = int(os.environ.get("IDOL_MAX_GAUSSIANS", "262144"))
OPTIMIZE = os.environ.get("IDOL_OPTIMIZE", "1") == "1"
WARMUP = os.environ.get("IDOL_WARMUP", "1") == "1"
IDLE_UNLOAD = int(os.environ.get("IDOL_IDLE_UNLOAD_SECONDS", "600"))  # 0 keeps the model loaded
JOBS_FILE = DATA / "jobs.json"

STEPS_MIN, STEPS_MAX = 2, 100
GAUSS_MIN = 32768
# Keep the cap valid even if someone sets an odd or too-small value.
MAX_GAUSSIANS = max(GAUSS_MIN, (MAX_GAUSSIANS // 32) * 32)
# Measured on an RTX 3080: splat count only moves file size and browser load, while sampler
# time is set by `steps`, so a preset has to move the two together to mean anything.
QUALITY = {
    "low": (25, min(32768, MAX_GAUSSIANS)),
    "medium": (50, min(262144, MAX_GAUSSIANS)),
    "high": (100, min(1048576, MAX_GAUSSIANS)),
}
ALLOWED_HOSTS = [h for h in os.environ.get("IDOL_ALLOWED_HOSTS", "").split(",") if h]

LIMITS = {
    "steps": {"min": STEPS_MIN, "max": STEPS_MAX, "default": 50},
    "num_gaussians": {"min": GAUSS_MIN, "max": MAX_GAUSSIANS,
                      "default": min(262144, MAX_GAUSSIANS), "multiple_of": 32},
    "cfg_strength": {"min": 0.0, "max": 10.0, "default": 3.0},
    "rescale_t": {"min": 0.5, "max": 10.0, "default": 3.0},
    "modes": ["synthetic", "real"],
    "default_mode": "real",
    "quality_presets": {k: {"steps": s, "num_gaussians": g} for k, (s, g) in QUALITY.items()},
}


# ------------------------------------------------------------------ engine ---

class Engine:
    """Holds one warm pipeline on the GPU, hands it out one call at a time, and gives the GPU
    back when no job has used it for IDLE_UNLOAD seconds."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.state = "starting"  # starting | loading | warming | ready | unloaded | error
        self.error: str | None = None
        self.ready_at: float | None = None  # stays set through unloads: jobs are accepted from here on
        self.last_used = time.monotonic()
        self.running_job: str | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            with self.lock:
                self._build()
        except Exception as exc:  # surfaced through /health and idol_status
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("pipeline failed to load")

    def _build(self) -> None:
        """Load and warm the pipeline. The caller holds the lock."""
        self.state = "loading"
        logger.info("loading pipeline from %s (optimize=%s)", MODEL_DIR, OPTIMIZE)
        from idol import IdolPipeline

        pipeline = IdolPipeline.from_pretrained(
            MODEL_DIR, device="cuda", dinov3_dtype="bf16", optimize=OPTIMIZE
        )
        self.state = "warming"
        if WARMUP:
            # The first compiled call is slow; pay for it here, not on a user's job.
            logger.info("warming up (triggers torch.compile)")
            pipeline(_blank_object(), mode="real", seed=0, steps=4, num_gaussians=GAUSS_MIN)
        self.pipeline = pipeline
        self.error = None
        self.ready_at = time.time()
        self.last_used = time.monotonic()
        self.state = "ready"
        logger.info("ready")

    def _reload(self) -> None:
        """Bring the pipeline back after an idle unload. The caller holds the lock.

        A failure leaves the state at unloaded, not error: VRAM is the likely cause, since
        unloading is what lets other work take it, and that clears up, so the next job retries."""
        t0 = time.time()
        failure = None
        try:
            self._build()
        except Exception as exc:
            logger.exception("pipeline failed to reload")
            failure = f"{type(exc).__name__}: {exc}"  # a string: the traceback pins the half-built model
        if failure:
            self.state, self.error = "unloaded", failure
            _release_memory()
            raise RuntimeError(f"model failed to reload: {failure}")
        logger.info("reloaded in %.1fs", time.time() - t0)

    @contextmanager
    def loaded(self):
        """One job's exclusive use of the pipeline. The idle clock restarts inside the lock, so
        the watcher cannot slip in between a long job ending and the clock moving."""
        with self.lock:
            try:
                if self.pipeline is None:
                    self._reload()
                yield self.pipeline
            finally:
                self.last_used = time.monotonic()

    def watch_idle(self) -> None:
        interval = max(5.0, min(IDLE_UNLOAD / 4, 60.0))
        logger.info("idle unload after %ds without a job", IDLE_UNLOAD)
        while True:
            time.sleep(interval)
            try:
                self._unload_if_idle()
            except Exception:
                logger.exception("idle unload failed")

    def _unload_if_idle(self) -> None:
        if not self.lock.acquire(blocking=False):
            return  # a job or a load has it, so this is not idle
        try:
            idle = time.monotonic() - self.last_used
            if self.pipeline is None or idle < IDLE_UNLOAD:
                return
            before = self.gpu_info().get("vram_used_mib")
            self.pipeline = None
            self.state = "unloaded"
            _release_memory()
            logger.info("unloaded after %.0fs idle (GPU %s -> %s MiB used)", idle, before,
                        self.gpu_info().get("vram_used_mib"))
        finally:
            self.lock.release()

    def gpu_info(self) -> dict[str, Any]:
        try:
            if not torch.cuda.is_available():
                return {"available": False}
            i = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info()
            return {
                "available": True,
                "name": torch.cuda.get_device_name(i),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
                "vram_total_mib": round(total / 2**20),
                "vram_used_mib": round((total - free) / 2**20),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}


def _release_memory() -> None:
    """Hand a dropped model back: VRAM to the driver, and host RAM to the OS, because glibc keeps
    what a load freed (measured 5-6 GB and climbing across reloads, against 2.2 GB trimmed).
    torch.compile's cache stays. It pins no weights and holds 13 MiB of RAM, ten reloads without
    clearing it stayed at full speed, and clearing it breaks seeds: a cold compile and a cached
    one give different output, so a process that compiled cold would shift on its first reload."""
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def _blank_object():
    """A soft grey blob on transparent ground - a real alpha channel, so the
    warmup skips BiRefNet entirely and just exercises the sampler and decoders."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([262, 262, 762, 762], fill=(150, 150, 155, 255))
    return img


engine = Engine()


# -------------------------------------------------------------------- jobs ---

JOBS: dict[str, dict[str, Any]] = {}
_jobs_guard = threading.Lock()


def _save_jobs() -> None:
    with _jobs_guard:
        keep = {}
        for k, v in JOBS.items():
            if v["state"] not in ("done", "trashed"):
                continue
            v = dict(v)
            inp = dict(v.get("input") or {})
            if inp.get("kind") == "base64":  # a data URI is megabytes; the object is the record
                inp["value"] = f"<{len(inp.get('value') or '')} chars of base64>"
            v["input"] = inp
            keep[k] = v
        JOBS_FILE.write_text(json.dumps(keep, indent=1), encoding="utf-8")


def _load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    try:
        for k, v in json.loads(JOBS_FILE.read_text(encoding="utf-8")).items():
            if (OBJECTS / f"{k}.ply").exists() or (TRASH / f"{k}.ply").exists():
                JOBS[k] = v
        logger.info("restored %d finished jobs", len(JOBS))
    except Exception:
        logger.exception("could not read %s", JOBS_FILE)


def _read_input(spec: dict[str, str]):
    """Accept an http(s) URL, a data: URI, or a path already on the server."""
    import io

    from PIL import Image

    kind, value = spec["kind"], spec["value"]
    if kind == "url":
        import requests

        r = requests.get(value, timeout=60)
        r.raise_for_status()
        return _loaded(Image.open(io.BytesIO(r.content)))
    if kind == "base64":
        import base64

        raw = value.split(",", 1)[1] if value.startswith("data:") else value
        return _loaded(Image.open(io.BytesIO(base64.b64decode(raw))))
    p = Path(value)
    if not p.is_absolute():
        p = DATA / value
    if not p.is_file():
        raise FileNotFoundError(f"no such file on the server: {p}")
    return _loaded(Image.open(p))


def _loaded(img):
    img.load()  # read the bytes now, so a closed socket cannot break the job later
    return img


def _classify(image: str) -> dict[str, str]:
    if image.startswith(("http://", "https://")):
        return {"kind": "url", "value": image}
    if image.startswith("data:"):
        return {"kind": "base64", "value": image}
    return {"kind": "path", "value": image}


def _check_params(steps: int, num_gaussians: int, cfg_strength: float, rescale_t: float) -> None:
    if not STEPS_MIN <= steps <= STEPS_MAX:
        raise ValueError(f"steps must be {STEPS_MIN}-{STEPS_MAX}")
    if not GAUSS_MIN <= num_gaussians <= MAX_GAUSSIANS:
        raise ValueError(f"num_gaussians must be {GAUSS_MIN}-{MAX_GAUSSIANS}")
    if num_gaussians % 32:
        raise ValueError("num_gaussians must be a multiple of 32")
    if not 0.0 <= cfg_strength <= 10.0:
        raise ValueError("cfg_strength must be 0-10")
    if not 0.5 <= rescale_t <= 10.0:
        raise ValueError("rescale_t must be 0.5-10")


def start_job(image: str, mode: str, seed: int, steps: int, num_gaussians: int,
              cfg_strength: float, rescale_t: float) -> str:
    _check_params(steps, num_gaussians, cfg_strength, rescale_t)
    if engine.state == "error":
        raise RuntimeError(f"model failed to load: {engine.error}")
    if engine.ready_at is None:
        raise RuntimeError(f"model is not ready yet ({engine.state}); wait for /health to report ready")
    engine.last_used = time.monotonic()  # the job may spend a while fetching its image first
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "state": "queued",
        "input": _classify(image),
        "params": {"mode": mode, "seed": seed, "steps": steps,
                   "num_gaussians": num_gaussians, "cfg_strength": cfg_strength,
                   "rescale_t": rescale_t},
        "created": time.time(),
        "error": None,
    }
    _save_jobs()
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job_id


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    job["state"] = "loading image"
    t0 = time.time()
    try:
        image = _read_input(job["input"])
        job["state"] = "waiting for model"
        with engine.loaded() as pipeline:
            engine.running_job = job_id
            job["state"] = "generating"
            started = time.time()
            out = pipeline(image, **job["params"])
            ply = OBJECTS / f"{job_id}.ply"
            out.gaussian.save_ply(str(ply))
            try:
                img = out.preprocessed_image
                img.save(OBJECTS / f"{job_id}-input.png")
                thumb = img.copy()
                thumb.thumbnail((200, 200))
                thumb.convert("RGB").save(OBJECTS / f"{job_id}-thumb.jpg", quality=82)
            except Exception:
                logger.exception("could not save preprocessed input")
        job.update(
            state="done",
            seconds=round(time.time() - started, 1),
            num_gaussians=int(out.gaussian.xyz.shape[0]),
            file=f"{job_id}.ply",
            input_image=f"{job_id}-input.png",
            camera=out.camera,
        )
        logger.info("job %s done in %.1fs (%d splats)", job_id, job["seconds"], job["num_gaussians"])
    except Exception as exc:
        job.update(state="error", error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - t0, 1))
        logger.exception("job %s failed", job_id)
    finally:
        engine.running_job = None
        _save_jobs()


# ------------------------------------------------------------------- links ---

def links(job: dict[str, Any], host: str | None) -> dict[str, str]:
    if PUBLIC_BASE:
        base = PUBLIC_BASE
    elif host:
        base = f"http://{host}"
    else:
        base = f"http://127.0.0.1:{PORT}"
    trashed = job["state"] == "trashed"
    return {
        "view_url": f"{base}/viewer/{job['id']}",
        "download_url": f"{base}/trash/{job['id']}.ply" if trashed else f"{base}/objects/{job['id']}.ply",
        "input_image_url": f"{base}/objects/{job['id']}-input.png" + ("?trash=1" if trashed else ""),
    }


def _public(job: dict[str, Any], host: str | None) -> dict[str, Any]:
    out = {k: v for k, v in job.items() if k not in ("input",)}
    if job["state"] in ("done", "trashed"):
        out.update(links(job, host))
    return out


def status_payload(host: str | None) -> dict[str, Any]:
    """Shared by /health and idol_status."""
    busy = ("queued", "loading image", "waiting for model", "generating")
    base = PUBLIC_BASE or (f"http://{host}" if host else f"http://127.0.0.1:{PORT}")
    return {
        "state": engine.state,
        "error": engine.error,
        # Whether jobs are accepted. An unloaded model is ready: the next job reloads it.
        "ready": engine.ready_at is not None,
        "model_loaded": engine.pipeline is not None,
        "seconds_idle": round(time.monotonic() - engine.last_used) if engine.pipeline is not None else None,
        "idle_unload_seconds": IDLE_UNLOAD,
        "gpu": engine.gpu_info(),
        "busy_with": engine.running_job,
        "queued": len([j for j in JOBS.values() if j["state"] in busy]),
        "objects_on_server": len([j for j in JOBS.values() if j["state"] == "done"]),
        "limits": LIMITS,
        "mcp_endpoint": f"{base}/mcp",
    }


def objects_payload(host: str | None, limit: int) -> dict[str, Any]:
    """Shared by /jobs and idol_objects."""
    done = [j for j in JOBS.values() if j["state"] == "done"]
    done.sort(key=lambda j: j.get("created", 0), reverse=True)
    return {"count": len(done), "objects": [_public(j, host) for j in done[:limit]]}


# --------------------------------------------------------------- MCP tools ---

from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver import Context  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.server.transport_security import TransportSecuritySettings  # noqa: E402

mcp = MCPServer(
    "mcp-image-to-3d",
    instructions=(
        "MCP Image to 3D turns a single image into a 3D Gaussian splat object using the Idol model. Generation is a background "
        "job that takes tens of seconds on a consumer GPU: call idol_generate_3d, then poll "
        "idol_job until state is done, and hand the user the view_url (interactive 3D in the "
        "browser) and download_url (.ply). Use wait_seconds to get the result in one call. "
        "After a while without jobs the model leaves the GPU to free VRAM: idol_status then shows "
        "state 'unloaded' with ready still true. That is normal, not a fault - just submit the "
        "job, which reloads the model first and sits in 'waiting for model' meanwhile. "
        "Check idol_status first if anything seems off."
    ),
)


def _host(ctx: Context) -> str | None:
    h = getattr(ctx, "headers", None) or {}
    return h.get("host") or h.get("Host")


@mcp.tool(title="Generate a 3D object from an image")
def idol_generate_3d(
    image: Annotated[str, Field(description="Source image: an http(s) URL, a data:image base64 URI, or a file path on the MCP Image to 3D server.")],
    ctx: Context,
    mode: Annotated[
        Literal["synthetic", "real"],
        Field(description="Crop margin, not image realism: 'real' crops loose (1.7x the object box), "
                          "'synthetic' crops tight (1.2x). 'real' is the default because it measured "
                          "better on every input tried, photographs and flat renders alike; use "
                          "'synthetic' only when the subject already fills the frame. This is the "
                          "biggest quality lever - see the description."),
    ] = "real",
    quality: Annotated[
        Literal["low", "medium", "high"],
        Field(description="steps and splat count as one choice: low = 25 steps/32768 splats "
                          "(~9-23 s, 2 MB, for trying out designs), medium = 50/262144 "
                          "(~18-44 s, 17 MB, default), high = 100/1048576 (~27-87 s, 68 MB, "
                          "final asset). The range is RTX 5090 to RTX 3080. Pass steps or "
                          "num_gaussians explicitly to move one of them on its own."),
    ] = "medium",
    steps: Annotated[int | None, Field(ge=STEPS_MIN, le=STEPS_MAX)] = None,
    num_gaussians: Annotated[int | None, Field(ge=GAUSS_MIN, le=MAX_GAUSSIANS)] = None,
    seed: int = 42,
    cfg_strength: Annotated[float, Field(ge=0.0, le=10.0)] = 3.0,
    rescale_t: Annotated[float, Field(ge=0.5, le=10.0)] = 3.0,
    tags: Annotated[list[str] | None, Field(max_length=12, description="Short labels that make "
                          "this object findable in the gallery search, e.g. ['sofa', 'living-room']. "
                          "They can also be added later in the web UI; they are the only thing the "
                          "search box reads besides the id and settings.")] = None,
    wait_seconds: Annotated[int, Field(ge=0, le=600)] = 0,
) -> dict[str, Any]:
    """Start building a 3D Gaussian splat object from one image and return its job id.

    `mode` sets the crop the model sees and nothing else: 'synthetic' frames the object so it
    fills 1/1.2 of the square, 'real' frames it at 1/1.7 (`CROP_MARGIN` in idol/preprocess.py).
    Your own framing is discarded, because the crop is rebuilt from the object's bounding box -
    feeding a tight image and an image with generous background produce the same object to within
    measurement noise (96.2% vs 96.4% well-defined surface on one sofa). So the only question is
    which scale the model reconstructs better, and measured on the reference GPU 'real' won on every
    input tried: at a fixed seed it took faint splats from 7.2% of all splats to 0.8% (well-defined
    surface 89% to 96%) and brought back four thin legs that 'synthetic' had replaced with a fog
    cloud; a flat-render fox also preferred it, 41% to 46% well-defined. The model card instead
    prescribes 'synthetic' for renders and generated images and 'real' for photographs - this
    service defaults to 'real' because of the measurements above, including on a render. Set mode
    explicitly if you would rather follow the card. If a result looks like it is floating on a
    cloud of faint splats, suspect the mode before blaming the model.

    The other defaults are measured too, and there is nothing to tune in them: a 10-run sweep at
    fixed splat count found no measurable difference from moving steps between 25 and 100,
    cfg_strength between 1 and 7, or rescale_t between 1 and 5 - the output geometry came out
    identical every time, and cfg_strength above 5 was slightly worse. Use `quality` to move
    between fast drafts and a final asset; leave cfg_strength and rescale_t alone.

    For the input image: keep the whole object inside the frame with clear margin, because the
    crop is cut from the object's own bounding box and anything outside the frame is simply
    gone. How much background you leave does not otherwise matter. Show the parts you need in
    3D - a straight-on view at the object's own height, so legs, undersides and other hidden
    structure read against the background, beat every angled or elevated view tested (97.6%
    well-defined surface, 0.5% faint, versus 82-89% and 7-13% for the same object shot from
    above at three quarters). Background colour only matters
    when the image has NO alpha channel: the pipeline then has to segment the object itself,
    and it does that far better against a plain light background than a dark one (13.2% faint
    on near-black against 6.7% on light grey). When a real alpha channel is present the
    background is thrown away regardless, because the SDK always composites onto black
    internally, and a supplied transparent cut-out then does as well as a light background.
    The image is always resized to 1024x1024, so a larger input buys nothing.

    num_gaussians is splat count, not scene detail. The geometry is fixed by the model's octree,
    so 524288 or 1048576 sample the same surface more densely and add mostly faint splats rather
    than new detail: measured on one input, 71% of splats sit above 0.1 opacity at 262144,
    falling to 49% at 1048576. Time and VRAM barely change, but the file quadruples (18 MB to
    71 MB) and browsers load it more slowly. 262144 is the default for that reason; raise it
    only for close-up inspection. More steps buys more quality than more splats does, and the
    input image is always cropped and resized to 1024x1024, so a larger input buys nothing."""
    try:
        q_steps, q_splats = QUALITY[quality]
        job_id = start_job(image, mode, seed, steps or q_steps, num_gaussians or q_splats,
                           cfg_strength, rescale_t)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except RuntimeError as exc:
        raise ToolError(str(exc)) from exc
    if tags:
        JOBS[job_id]["tags"] = _clean_tags(tags)
        _save_jobs()

    deadline = time.time() + wait_seconds
    while wait_seconds and JOBS[job_id]["state"] in ("queued", "loading image", "waiting for model", "generating") and time.time() < deadline:
        time.sleep(2)
    if JOBS[job_id]["state"] == "error":
        raise ToolError(f"generation failed: {JOBS[job_id]['error']}")
    return _public(JOBS[job_id], _host(ctx))


@mcp.tool(title="Check a 3D generation job")
def idol_job(
    job_id: str,
    ctx: Context,
) -> dict[str, Any]:
    """Return the status of a job from idol_generate_3d. When state is 'done' this includes
    view_url (interactive 3D in a browser), download_url (.ply) and the splat count."""
    job = JOBS.get(job_id)
    if not job:
        raise ToolError(f"no job with id {job_id}. Call idol_objects to see what is on the server.")
    return _public(job, _host(ctx))





@mcp.tool(title="List generated 3D objects")
def idol_objects(
    ctx: Context,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
) -> dict[str, Any]:
    """List finished 3D objects on this server, newest first, each with its view and download
    links. Use this to reuse an object instead of generating again."""
    return objects_payload(_host(ctx), limit)


@mcp.tool(title="MCP Image to 3D server status")
def idol_status(ctx: Context) -> dict[str, Any]:
    """Whether the model is loaded and warm, which GPU it is on, free VRAM, the queue, and the
    legal ranges for every generation parameter. state 'unloaded' means the model left the GPU
    after sitting idle; the next job reloads it by itself, so there is nothing to fix."""
    return status_payload(_host(ctx))


# -------------------------------------------------------------------- HTTP ---

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.routing import Mount  # noqa: E402

WEB = Path(__file__).resolve().parent.parent / "web"

# No auth on a trusted LAN or tailnet, and the service is reachable under several names
# (LAN IP, tailnet IP, DNS name). An exact allowlist would answer 421 to any address not
# listed, including legitimate ones - so it is opt-in via IDOL_ALLOWED_HOSTS.
if ALLOWED_HOSTS:
    mcp_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=ALLOWED_HOSTS
    )
else:
    mcp_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

# Built before the FastAPI app on purpose: the session manager is created lazily, only
# when streamable_http_app() is called, and the lifespan below accesses it.
mcp_app = mcp.streamable_http_app(transport_security=mcp_security)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        OBJECTS.mkdir(parents=True, exist_ok=True)
        TRASH.mkdir(parents=True, exist_ok=True)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        _load_jobs()
        threading.Thread(target=engine.load, daemon=True).start()
        if IDLE_UNLOAD > 0:
            threading.Thread(target=engine.watch_idle, name="idle-unload", daemon=True).start()
        yield


app = FastAPI(title="MCP Image to 3D", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def cross_origin_isolation(request: Request, call_next):
    """The browser splat viewer parses in a worker using SharedArrayBuffer, which the
    browser only allows on a cross-origin-isolated page. Everything the viewer loads is
    same-origin, so these two headers are enough to enable it."""
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# The viewer's three.js and splat renderer are served from here rather than from a
# CDN, so a client on a private network with no internet still gets a working 3D view.
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")
app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


def _request_host(request: Request) -> str:
    return request.headers.get("host") or f"127.0.0.1:{PORT}"


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    return status_payload(_request_host(request)) | {"jobs": len(JOBS)}


@app.post("/jobs")
def create_job(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    image = body.get("image") or body.get("image_url") or body.get("image_base64")
    if not image:
        raise HTTPException(400, "send 'image': a URL, a data: URI, or a server-side path")
    p = body.get("params", {})
    quality = body.get("quality", "medium")
    if quality not in QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(QUALITY)}")
    q_steps, q_splats = QUALITY[quality]
    try:
        job_id = start_job(
            image,
            p.get("mode", body.get("mode", "real")),
            int(p.get("seed", body.get("seed", 42))),
            int(p.get("steps", body.get("steps", q_steps))),
            int(p.get("num_gaussians", body.get("num_gaussians", q_splats))),
            float(p.get("cfg_strength", body.get("cfg_strength", 3.0))),
            float(p.get("rescale_t", body.get("rescale_t", 3.0))),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    # Labels given at creation are stored with the job: they are what the gallery
    # search reads, so a library built from chat stays findable.
    if body.get("tags"):
        JOBS[job_id]["tags"] = _clean_tags(body["tags"])
        _save_jobs()
    return _public(JOBS[job_id], _request_host(request))


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"no job {job_id}")
    return _public(job, _request_host(request))


@app.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    return objects_payload(_request_host(request), 100)


def _object_response(folder: Path, fname: str):
    folder = folder.resolve()
    safe = (folder / fname).resolve()
    if not safe.is_relative_to(folder) or safe == folder or not safe.is_file():
        raise HTTPException(404, f"no file {fname}")
    media = {".ply": "application/octet-stream", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    return FileResponse(safe, media_type=media.get(safe.suffix, "image/png"))


@app.get("/objects/{fname}")
def get_object(fname: str, trash: int = 0):
    return _object_response(TRASH if trash else OBJECTS, fname)


@app.get("/trash/{fname}")
def get_trash_object(fname: str):
    # A path without a query string, because the splat loader reads the extension.
    return _object_response(TRASH, fname)


@app.delete("/objects/{job_id}")
def delete_object(job_id: str):
    """Remove an object, its saved input image and its thumbnail, and forget the job."""
    if not job_id.isalnum():
        raise HTTPException(400, f"bad object id {job_id}")
    if job_id not in JOBS:
        raise HTTPException(404, f"no object {job_id}")
    for folder in (OBJECTS, TRASH):
        for suffix in (".ply", "-input.png", "-thumb.jpg"):
            (folder / f"{job_id}{suffix}").unlink(missing_ok=True)
    JOBS.pop(job_id)
    _save_jobs()
    logger.info("object %s deleted", job_id)
    return {"deleted": job_id}


@app.get("/viewer/{job_id}")
def viewer(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job or job["state"] not in ("done", "trashed"):
        raise HTTPException(404, f"no finished object {job_id}")
    html = (WEB / "viewer.html").read_text(encoding="utf-8")
    ply = f"/trash/{job['id']}.ply" if job["state"] == "trashed" else f"/objects/{job['id']}.ply"
    meta = json.dumps({
        "ply": ply,
        "id": job["id"],
        "splats": job.get("num_gaussians"),
        "params": job.get("params"),
        "seconds": job.get("seconds"),
    })
    return HTMLResponse(html.replace("__META__", meta),
                        headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ gallery --
# The web gallery is a normal page that fetches its data, so it can refresh while
# jobs finish.  It follows the Image Gen media gallery: same look, sort, filters,
# multi-select and trash.

SUFFIXES = (".ply", "-input.png", "-thumb.jpg")


def _quality_of(job: dict[str, Any]) -> str:
    """Which preset produced this object, or 'custom' when the settings were moved by hand."""
    steps = (job.get("params") or {}).get("steps")
    for name, (preset_steps, preset_splats) in QUALITY.items():
        if steps == preset_steps and job.get("num_gaussians") == preset_splats:
            return name
    return "custom"


def _gallery_item(job: dict[str, Any], folder: Path) -> dict[str, Any]:
    ply = folder / f"{job['id']}.ply"
    params = job.get("params") or {}
    return {
        "id": job["id"],
        "created": job.get("created"),
        "deleted": job.get("deleted"),
        "seconds": job.get("seconds"),
        "splats": job.get("num_gaussians"),
        "quality": _quality_of(job),
        "mode": params.get("mode"),
        "steps": params.get("steps"),
        "seed": params.get("seed"),
        "bytes": ply.stat().st_size if ply.is_file() else 0,
        "thumb": (folder / f"{job['id']}-thumb.jpg").is_file(),
        "tags": sorted(job.get("tags") or []),
        "camera": job.get("camera"),
    }


def _move_objects(ids: list[str], src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    moved = []
    for oid in ids:
        if not oid.isalnum() or oid not in JOBS:
            continue
        for suffix in SUFFIXES:
            f = src / f"{oid}{suffix}"
            if f.is_file():
                f.replace(dst / f.name)
        moved.append(oid)
    return moved


def _clean_tags(values: Any) -> list[str]:
    """Normalise labels to a sorted list of short lowercase tags."""
    out = set()
    for value in values or []:
        tag = str(value).strip().lower()
        if tag and len(tag) <= 40:
            out.add(tag)
    return sorted(out)


def _tag_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in JOBS.values():
        if job["state"] != "done":
            continue
        for tag in job.get("tags") or ():
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@app.get("/api/objects")
def api_objects() -> dict[str, Any]:
    done = [j for j in JOBS.values() if j["state"] == "done"]
    done.sort(key=lambda j: j.get("created", 0), reverse=True)
    return {"items": [_gallery_item(j, OBJECTS) for j in done],
            "tags": _tag_counts(),
            "trash": sum(1 for j in JOBS.values() if j["state"] == "trashed")}


@app.get("/api/trash")
def api_trash() -> dict[str, Any]:
    gone = [j for j in JOBS.values() if j["state"] == "trashed"]
    gone.sort(key=lambda j: j.get("deleted") or 0, reverse=True)
    return {"items": [_gallery_item(j, TRASH) for j in gone]}


@app.post("/api/trash")
def api_to_trash(body: dict[str, Any]) -> dict[str, Any]:
    moved = _move_objects([str(i) for i in body.get("ids", [])], OBJECTS, TRASH)
    for oid in moved:
        JOBS[oid].update(state="trashed", deleted=time.time())
    _save_jobs()
    return {"moved": moved}


@app.post("/api/restore")
def api_restore(body: dict[str, Any]) -> dict[str, Any]:
    moved = _move_objects([str(i) for i in body.get("ids", [])], TRASH, OBJECTS)
    for oid in moved:
        JOBS[oid]["state"] = "done"
        JOBS[oid].pop("deleted", None)
    _save_jobs()
    return {"restored": moved}


@app.post("/api/purge")
def api_purge(body: dict[str, Any]) -> dict[str, Any]:
    ids = [str(i) for i in body.get("ids", []) if str(i).isalnum()] or [
        k for k, v in JOBS.items() if v["state"] == "trashed"]
    for oid in ids:
        for suffix in SUFFIXES:
            (TRASH / f"{oid}{suffix}").unlink(missing_ok=True)
            (OBJECTS / f"{oid}{suffix}").unlink(missing_ok=True)
        JOBS.pop(oid, None)
    _save_jobs()
    return {"deleted": ids}


@app.post("/api/tags")
def api_tags(body: dict[str, Any]) -> dict[str, Any]:
    """Add or remove tags on one object. Tags are what make the search box useful.

    The merge happens here rather than on the page, so two edits typed quickly both
    land instead of the second overwriting a list the first one had just changed.
    """
    oid = str(body.get("id", ""))
    job = JOBS.get(oid)
    if not oid.isalnum() or not job:
        raise HTTPException(404, f"no object {oid}")

    def clean(values) -> set[str]:
        return set(_clean_tags(values))

    tags = set(job.get("tags") or ())
    if body.get("tags") is not None:  # explicit replace
        tags = clean(body["tags"])
    tags |= clean(body.get("add"))
    tags -= clean(body.get("remove"))
    job["tags"] = sorted(tags)
    _save_jobs()
    return {"id": oid, "tags": job["tags"]}


@app.post("/api/upload")
async def api_upload(request: Request) -> dict[str, Any]:
    """Store an image the page wants generated, and return the path to pass to /jobs."""
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    if len(body) > 40_000_000:
        raise HTTPException(413, "keep uploads under 40 MB")
    suffix = Path(request.query_params.get("name", "")).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        suffix = ".png"
    name = f"{uuid.uuid4().hex[:12]}{suffix}"
    (UPLOADS / name).write_bytes(body)
    return {"path": f"uploads/{name}", "bytes": len(body)}


@app.post("/api/zip")
def api_zip(body: dict[str, Any]):
    """Bundle several .ply files into one download. Stored, not deflated: these are
    float streams that barely shrink, and the GPU box should not spend CPU on it."""
    plys = []
    for i in body.get("ids", []):
        if not str(i).isalnum():
            continue
        for folder in (OBJECTS, TRASH):
            p = folder / f"{i}.ply"
            if p.is_file():
                plys.append((p, f"{i}.ply"))
                break
    if not plys:
        raise HTTPException(404, "nothing to download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for path, name in plys:
            z.write(path, name)
    buf.seek(0)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="idol-objects.zip"'})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


app.routes.append(Mount("/", app=mcp_app))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
