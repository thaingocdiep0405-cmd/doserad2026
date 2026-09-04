from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
import uvicorn
from uvicorn.config import LOGGING_CONFIG

import inference


MODEL_BUNDLE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_BUNDLE
    MODEL_BUNDLE = inference.load_model()
    yield
    MODEL_BUNDLE = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return Response(
        status_code=status.HTTP_200_OK if MODEL_BUNDLE is not None else status.HTTP_404_NOT_FOUND
    )


@app.post("/invoke")
async def invoke():
    if MODEL_BUNDLE is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    inference.run(MODEL_BUNDLE)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
