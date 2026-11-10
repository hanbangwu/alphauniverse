Site behaviour
==============

The web app's exact behaviour. :doc:`/using` is the visitor's summary.

Password
--------

- A page request without the ``password`` cookie redirects to ``/login``, and the app makes no API request. The server hook checks only that the cookie exists.
- A correct password opens the map. A wrong one shows **Incorrect password.** Any other failure shows **The server is unavailable. Try again in a moment.**
- A ``401`` from an SDK call returns to ``/login``.

Layout
------

- A glass dock at the top holds the logo, the tabs, a theme toggle and icon links to GitHub and ``/docs``, each icon named in a tooltip on hover.
- The UMAP controls sit in a non-modal drawer on the left, open while the UMAP tab shows unless hidden with its button. The open galaxy shows in a non-modal drawer on the right; closing it closes the galaxy. Neither closes on a click outside or by dragging.

Sky
---

- The default tab. Tiles are Legacy Survey DR10 south JPEGs from ``legacysurvey.org``, requested by the browser, native to zoom 14 and enlarged to zoom 17. The map, like the UMAP, fills the window under the drawers. It opens fitted to the dataset's extent plus one cutout width, pans and zooms freely across the survey from there, and keeps north up. A scale bar in the bottom left gives angular size in degrees, arcminutes or arcseconds, and above it the RA and Dec under the cursor show to five decimal places, RA wrapped to 0° to 360°.
- ``/positions`` gives each galaxy's ``ra`` and ``dec``, drawn at longitude 180° − RA and latitude Dec on the tiles' Web Mercator grid.
- Above zoom 12 and below zoom 16, every galaxy is drawn once ``/positions`` loads as a white 5 px dot whose copy grows to twice its radius on a cubic ease-out and fades every 1 s, all in step; with reduced motion set, the dots stay still. At zoom 12 and below, only the open galaxy is drawn, as a still dot, and none can be hovered or clicked. The colour and range controls show only on the UMAP tab.
- While dots show, hover and click take the nearest galaxy within 6 px. Hover rings that dot in red and shows the galaxy's image and name. Click opens the galaxy and starts no search. The open galaxy's dot stays ringed.
- From zoom 16, the deepest being 17, dots hide and every galaxy has its 25″ Legacy Survey cutout drawn as a 24 × 24 grid, the open galaxy's brighter and on top. Hover and click take the open galaxy when its grid holds the cursor, else the nearest galaxy whose grid does. Hover highlights the patch; click opens the galaxy and toggles the patch in the Legacy Survey image selection of the search dialog, which only the open galaxy can hold. Patches count from the top left of the north-up, east-left image.
- A click that takes no galaxy, at any zoom, closes the open galaxy and clears its selection and search.

UMAP
----

- **Any position**, the default, draws ``any_position_points``; **Same place** draws ``same_place_points``.
- A property is coloured on 32 viridis bins from its 1st to its 99th percentile; values beyond either end take that end's colour. Redshift is the default. g − r is only measured.
- The legend gives the range, its unit and how many galaxies have a measured value, a prediction or none.
- The slider moves in 200 steps across the legend's range and counts the galaxies in range. A thumb left at its end also keeps the values beyond that side, and the readout then shows only the other bound. The range resets when the property or source changes. The map circles only search matches in range.
- Box and lasso selections keep only galaxies in range, list them by object ID, each clickable (an empty selection shows nothing), and export their ``object_id``, ``ra`` and ``dec`` as CSV. A selection belongs to the map it was drawn on and drops when the layout changes or on a click elsewhere on the map.
- Hover shows the galaxy's image and property, with the prediction's width when the value is AION's.

Galaxy
------

- Images are the central 96 × 96 px, coloured from their *g r i z* flux by the Legacy Survey's mapping.
- Spectra are smoothed for display with a Gaussian of σ = 2 pixels. Search uses AION's predictions, never the smoothed flux.
- Groups collapse, and stay open or collapsed for every galaxy in both views. The search view adds the catalogue group: Galaxy Zoo 10, PROVABGS and the redshift, with the 26 values AION reads selectable.
- **AION predictions** shows the :term:`AION predictions <AION prediction>` of every survey, observed or not: images in place of the observed ones, coloured the same way and marked AION; spectra over the observed ones, each scaled to its flux, or alone and normalised; in search, the median of the predicted distribution beside each of the 25 catalogue values and the redshift AION reads, or beside DESI's when the galaxy has no redshift AION reads. In the search, an AION prediction of a survey the galaxy lacks cannot be selected.

Similarity search
-----------------

- A click or tap toggles a cell or window. A mouse drag selects every one it passes over, or deselects them if it starts on a selected one. On touch, a sideways drag over an image does the same and a vertical swipe scrolls. With the keyboard, Tab to an image or spectrum, move with the arrow keys and press Enter or Space; a zoomed spectrum pans to follow.
- A search returns 1 to 128 galaxies, 32 by default. **Any position** is set separately for images and spectra.
- Each match shows its similarity, a note when it lacks a selected observation and was matched on AION's prediction, and a map for each image and spectrum it has, or for every one with AION predictions on.
- **Mask** has a threshold for the query and one for the matches for images and for spectra, and one for the query's catalogue values, each by default the middle of its maps' range. **Invert** keeps the colour below the threshold instead.
- **Export as ECSV** saves the query galaxy, then the matches in rank order, as ``alphauniverse-search.ecsv``: ``object_id``, ``ra``, ``dec``, ``similarity`` (``score`` for whole modes), one array column per image (``float32[24,24]``, by row then column) and spectrum (``float32[272]``), and one column per catalogue value, named after it. A map is empty where the galaxy lacks the observation, whatever **AION predictions** shows. ``meta`` holds the request, the selection as grid cells, wavelength ranges in Å and catalogue names, and the page's address. It is disabled while a new search loads.
- Clicking a match's object ID selects it with an empty selection. After a search the map circles the query and its matches in range until another galaxy is selected or another search runs.
- **Whole modes**: a click on an image, a spectrum or a catalogue section (Redshift, Legacy Survey, HSC) toggles that mode, outlined. **Any position** applies to the chosen images and spectra. A mode the galaxy lacks can be chosen when AION predictions show it; the search reads AION's prediction and says so. A match shows its score, one signed bar per mode for its contribution, and the same-place map of every slot. The cell, window and value selection is kept and returns when the switch is off; one search never mixes the two.
- Each map is coloured on its own scale, from its lowest to its highest value; hover reads a value.

Links
-----

- The address records the view: ``projection``, ``property``, ``source``, ``range``, ``region``, ``galaxy``, ``explorer``, ``predictions``, ``whole``, ``layer``, ``mask``, ``invert``, the five mask thresholds, ``tab`` and ``text``. A value at its default or failing validation is ignored and left out. Lists, ``range`` and ``region`` are JSON. The address updates on each change, a slider's on release, and adds no history entry.
- The last similarity search is recorded in the ``/search`` or ``/search/modes`` parameters it sent. Opening the address restores its selection and runs it again. A selection changed but not searched is not recorded.
- Collapsed groups, the theme and the disclaimer stay in the browser's storage, not the address.

Text search
-----------

- A query has up to 500 characters. The 32 matches each show the galaxy, the modality it matched by and its cosine similarity.
- The export saves the matches in rank order as CSV with ``object_id``, ``ra``, ``dec`` and ``cosine_similarity``. It is disabled while a new query loads.
