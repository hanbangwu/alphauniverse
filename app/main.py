from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI

from app.config import GRAM_NAMES, MODES, OBSERVATIONS, galaxy_count
from app.dataset import routes as dataset_routes
from app.encoding import routes as encoding_routes
from app.encoding.store import image_lattice, observed, spectrum_lattice
from app.login import routes as login_routes
from app.login.password import require_password
from app.predictions import routes as predictions_routes
from app.predictions.store import aion_predictions, basis, predictions
from app.predictions.whole_mode import stored
from app.projection import routes as projection_routes
from app.search import routes as search_routes
from app.text_search import routes as text_search_routes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    galaxy_count()
    predictions()
    basis()
    image_lattice()
    spectrum_lattice()
    for column in OBSERVATIONS:
        observed(column)
    for mode in MODES:
        aion_predictions(mode)
    for name in GRAM_NAMES:
        stored(name)
    yield


app = FastAPI(
    title="alphauniverse",
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
    dependencies=[Depends(require_password)],
)
app.include_router(login_routes.router)
app.include_router(projection_routes.router)
app.include_router(dataset_routes.router)
app.include_router(encoding_routes.router)
app.include_router(predictions_routes.router)
app.include_router(search_routes.router)
app.include_router(text_search_routes.router)
