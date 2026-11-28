"""Queue-only API; inference and visualization never cross this browser boundary."""

from contextlib import asynccontextmanager
import hmac
from pathlib import Path
import secrets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from core.config import ROOT
from core.store import Conflict


class QueueRequest(BaseModel):
    loan_id: str = Field(min_length=1, max_length=64)
    engines: list[str] = Field(min_length=1, max_length=3)


class CommandRequest(BaseModel):
    command_id: str = Field(min_length=1, max_length=128)
    name: str
    confirmed: bool = False


def create_app(controller):
    @asynccontextmanager
    async def lifespan(app):
        controller.run()
        yield
        controller.shutdown()

    app = FastAPI(
        title="EMBSYS SOP queue",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    token = controller.settings.token

    @app.middleware("http")
    async def protect(request: Request, call_next):
        # Host validation blocks DNS rebinding; no CORS and JSON-only mutations block cross-site forms.
        from urllib.parse import urlsplit

        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        if controller.settings.host in (
            "127.0.0.1",
            "localhost",
            "::1",
        ) and host not in ("127.0.0.1", "localhost", "::1"):
            return JSONResponse({"detail": "Invalid host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            auth = request.headers.get("authorization", "")
            if token and not hmac.compare_digest(
                auth.encode("utf-8"), ("Bearer " + token).encode("utf-8")
            ):
                return JSONResponse(
                    {"detail": "Authentication required"}, status_code=401
                )
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Cross-origin requests are disabled"}, status_code=403
                )
            if (
                request.method == "POST"
                and "application/json" not in request.headers.get("content-type", "")
            ):
                return JSONResponse(
                    {"detail": "JSON content type required"}, status_code=415
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "frontend/index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in ("app.js", "style.css"):
            raise HTTPException(404)
        return FileResponse(ROOT / "frontend" / name)

    @app.get("/api/state")
    def state():
        result = controller.state()
        # The browser is a scheduler, not an evidence/debug viewer.
        safe = {
            k: v
            for k, v in result.items()
            if k in ("state", "current", "next", "queue", "enabled", "mode")
        }
        allowed = (
            "id",
            "loan_id",
            "session_id",
            "engine",
            "position",
            "status",
            "attempt_id",
        )
        for key in ("current", "next"):
            safe[key] = (
                {k: v for k, v in (result[key] or {}).items() if k in allowed}
                if result[key]
                else None
            )
        started_sessions = {
            row["session_id"]
            for row in result["queue"]
            if row.get("linked") and row.get("attempt_id")
        }
        safe["queue"] = []
        for row in result["queue"]:
            removable = (
                row["status"] == "PENDING"
                and not row.get("attempt_id")
                and row["session_id"] not in started_sessions
            )
            if (
                row.get("linked")
                and row["engine"] == "purity"
                and any(
                    sibling["session_id"] == row["session_id"]
                    and sibling["engine"] == "packet"
                    and sibling["status"] != "CANCELLED"
                    for sibling in result["queue"]
                )
            ):
                removable = False
            safe["queue"].append(
                dict(
                    {k: v for k, v in row.items() if k in allowed}, can_remove=removable
                )
            )
        return safe

    @app.get("/api/hardware")
    def hardware():
        return controller.state()

    @app.post("/api/queue", status_code=201)
    def enqueue(body: QueueRequest):
        return controller.enqueue(body.loan_id.strip(), body.engines)

    @app.delete("/api/queue/{activity_id}")
    def remove(activity_id: str):
        controller.remove(activity_id)
        return {"removed": activity_id}

    @app.post("/api/commands")
    def command(body: CommandRequest):
        return controller.command(body.command_id, body.name, body.confirmed)

    return app
