"""Expose artifacts ZIP via a tiny FastAPI download server."""

from __future__ import annotations
import os

from quantacap.utils.optional_import import optional_import


ART = "artifacts"
ZIP_NAME = "quion_experiment.zip"
ZIP_PATH = os.path.join(ART, ZIP_NAME)


def _build_app():
    fastapi_mod = optional_import(
        "fastapi",
        pip_name="fastapi",
        purpose="serve the download endpoint",
    )
    responses_mod = optional_import(
        "fastapi.responses",
        pip_name="fastapi",
        purpose="serve the download endpoint",
    )
    FastAPI = fastapi_mod.FastAPI
    HTTPException = fastapi_mod.HTTPException
    FileResponse = responses_mod.FileResponse
    PlainTextResponse = responses_mod.PlainTextResponse

    app = FastAPI(title="Quantacap Download")

    @app.get("/", response_class=PlainTextResponse)
    def root() -> str:
        return (
            "Quantacap download server\n"
            f"GET /download -> {ZIP_NAME}\n"
        )

    @app.get("/download")
    def download():
        if not os.path.isfile(ZIP_PATH):
            raise HTTPException(
                status_code=404,
                detail=f"Missing {ZIP_PATH}. Run zip_artifacts first.",
            )
        return FileResponse(
            ZIP_PATH,
            filename=ZIP_NAME,
            media_type="application/zip",
        )

    return app


def serve(*, port: int = 8009, host: str = "0.0.0.0") -> None:
    uvicorn_mod = optional_import(
        "uvicorn",
        pip_name="uvicorn",
        purpose="run the download server",
    )
    app = _build_app()
    uvicorn_mod.run(app, host=host, port=port)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


if __name__ == "__main__":
    serve(
        port=_env_int("QUANTACAP_PORT", 8009),
        host=os.environ.get("QUANTACAP_HOST", "0.0.0.0"),
    )
