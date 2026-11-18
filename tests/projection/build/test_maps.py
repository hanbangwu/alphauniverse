from pathlib import Path

from app.config import MAPS, MODES, galaxy_count
from app.projection.build import maps
from app.search import pql

NEIGHBORS = 5


def test_each_galaxys_map_neighbors_are_its_whole_mode_search_matches(
    tree: Path,
) -> None:
    for anywhere in MAPS.values():
        found = maps.nearest(galaxy_count(), anywhere, NEIGHBORS)[1]
        for galaxy in range(galaxy_count()):
            searched = pql.mode_search(
                pql.ModeQuery(
                    galaxy=galaxy,
                    modes=set(MODES),
                    anywhere=anywhere,
                    matches=NEIGHBORS,
                )
            )
            assert found[galaxy].tolist() == searched.galaxies[1:].tolist()
