Using the site
==============

Browse the map
--------------

Each point is a galaxy, placed near the galaxies AION finds most alike (:doc:`methods/map`).

- **Any position** groups galaxies whose features match wherever they sit in the image or spectrum. **Same place** groups them only when the features are in the same place.
- Colour the points by redshift, stellar mass, star formation rate, specific star formation rate or g − r colour. **Measured first** uses the measured value where there is one and AION's prediction, faded, elsewhere. **Predicted** uses AION's prediction for every galaxy.
- Drag the legend's slider to show only the galaxies in a range, for example 0.3 < z < 0.5.
- Draw a box or a lasso to select galaxies, then open any of them or export the selection as CSV.
- Hover over a point to preview the galaxy. Click it to open it.

Look at a galaxy
----------------

Galaxies are named by their Legacy Survey DR10 object ID. A galaxy shows its position, linked to its Legacy Survey cutout, the surveys it was matched in, its Legacy Survey and HSC images and its DESI and SDSS spectra. Spectra are smoothed for display only. Zoom a spectrum with the scroll wheel or a pinch, and pan with Shift-drag or one finger.

See what AION predicts
----------------------

The **AION predictions** switch shows the galaxy's :term:`templates <template>`: what AION expects it to look like in every survey, given the data it has. A galaxy without a DESI spectrum still gets a predicted one. Predicted images replace the observed ones, predicted spectra are drawn over the observed ones, and predicted catalogue values appear beside the measured ones. AION does not predict a spectrum's overall brightness, so a predicted spectrum with no observed one beside it is normalised. A template is AION's expectation, not a measurement.

Find similar galaxies
---------------------

Open a galaxy, start a search and pick what should match: image cells, spectrum windows, catalogue values, or any mix. Click a cell or window, drag across several, or use the keyboard (Tab, arrow keys, Enter).

- **Any position** matches a feature wherever it sits, for example an emission line at another redshift. Without it, matches must have the feature in the same place.
- Each match shows a similarity from 0 to 1 and a map of where it agrees with your galaxy. **Mask** fades the parts below a threshold.
- A match that lacks an observation you selected is ranked on AION's prediction of it, and says so.
- Click a match's ID to search from it.
- **Export as ECSV** saves your galaxy and its matches as an :term:`ECSV` table: ID, position, similarity, and each map, empty where a galaxy lacks that observation. The header records what you selected and the page's address.
- **Whole modes** compares entire observations instead: click an image, a spectrum or a group of catalogue values. Each match then shows how much each observation contributed.
- The **This galaxy** and **Population** maps shade your galaxy to show what a selection picks out (:doc:`methods/query-maps`).

:doc:`methods/similarity-search` explains the ranking.

Search by text
--------------

Describe a galaxy in a sentence. The site returns the 32 closest matches by image or by spectrum, so a galaxy can appear twice (:doc:`methods/text-search`). Export the matches as CSV with their object ID, position and cosine similarity.

Share a view
------------

The address in the browser records what you see: the map's layout, colour, range and selection, the open galaxy, the last search and its display settings, and the text search. Copy it to return to the same view or to send it to someone.

Download the data
-----------------

Every galaxy's AION embeddings, tokens and properties (redshift, stellar mass, star formation rates and g − r colour, measured and predicted) can be downloaded. :doc:`dev/reference` describes the files.
