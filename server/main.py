from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI

from alphauniverse.config import GRAM_NAMES, MODES, OBSERVATIONS, galaxy_count
from alphauniverse.encoding.store import object_ids, observed
from alphauniverse.predictions.store import aion_predictions, basis, predictions
from alphauniverse.predictions.whole_mode import stored
from server.dataset import routes as dataset_routes
from server.encoding import routes as encoding_routes
from server.encoding.false_color import image_lattice, spectrum_lattice
from server.login import routes as login_routes
from server.login.password import require_password
from server.predictions import routes as predictions_routes
from server.projection import routes as projection_routes
from server.search import routes as search_routes
from server.text_search import routes as text_search_routes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    galaxy_count()
    predictions()
    basis()
    image_lattice()
    spectrum_lattice()
    object_ids()
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
