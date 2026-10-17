from pathlib import Path

from app import maps, pql
from app.config import MAPS, MODES, galaxy_count

NEIGHBOURS = 5


def test_each_galaxys_map_neighbours_are_its_whole_mode_search_matches(
    tree: Path,
) -> None:
    for anywhere in MAPS.values():
        found = maps.nearest(galaxy_count(), anywhere, NEIGHBOURS)[1]
        for galaxy in range(galaxy_count()):
            searched = pql.mode_search(
                pql.ModeQuery(
                    galaxy=galaxy,
                    modes=set(MODES),
                    anywhere=anywhere,
                    matches=NEIGHBOURS,
                )
            )
            assert found[galaxy].tolist() == searched.galaxies[1:].tolist()
