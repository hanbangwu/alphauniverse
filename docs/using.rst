Using the site
==============

This page says what you can do on the site and what each result means. The methods behind each feature are in the Methods pages.

Browse the map
--------------

The map shows every galaxy, or every token, placed by :doc:`methods/map`. Switch between **Mean**, one point per galaxy, and **Full**, one point per token embedding of every galaxy. Points are coloured by Galaxy Zoo 10 morphology; most galaxies have no label. Filter by morphology to show only some classes. Hover over a point to see the galaxy's image and label, and click it to select the galaxy; in Full, any of a galaxy's tokens selects it.

Look at a galaxy
----------------

A selected galaxy shows which surveys it is matched in and, in groups that can each be collapsed:

- its Legacy Survey and HSC images, the central 96 × 96 px of each: the Legacy Survey colour image, and the HSC image coloured from its *g r i z* flux by the mapping that made the Legacy Survey's;
- its DESI and SDSS spectra, smoothed for display with a Gaussian of σ = 2 pixels; search uses AION's predictions, never the smoothed flux.

Every spectrum zooms with the wheel or a pinch and pans with Shift-drag or a one-finger drag.

The search shows the same groups and a third, its catalogue values, with Galaxy Zoo 10, PROVABGS and the redshift; the 26 values AION reads can be selected for search. A group stays open or collapsed for every galaxy, in both views.

**AION predictions**, a switch beside the galaxy and in the search, shows the galaxy's :term:`templates <template>` in every survey, observed or not:

- AION's Legacy Survey and HSC images in place of the observed ones, coloured the same way and marked AION;
- AION's DESI and SDSS spectra over the observed ones, each scaled to its flux; without the observed spectrum, AION's alone, normalised, since AION does not predict a spectrum's overall flux;
- in the search, the median of AION's predicted distribution beside each of the 25 catalogue values it reads, and beside the redshift it reads, or beside DESI's when the galaxy has no redshift AION reads.

A template is what AION expects from the galaxy's tokens (:doc:`concepts`), not a measurement. In the search, a template of a survey the galaxy lacks cannot be selected.

Search by similarity
--------------------

Start a search from the selected galaxy and pick what to compare: cells of its Legacy Survey or HSC image, windows of its DESI or SDSS spectrum, catalogue values, or any mix. Click or tap a cell or a window to select or deselect it, or drag the mouse across several: the drag selects every one it passes over, or deselects them if it starts on a selected one. On a touch screen, a sideways drag over an image does the same, and a vertical swipe scrolls. With the keyboard, Tab to an image or a spectrum, move with the arrow keys and press Enter or Space; a zoomed spectrum pans to follow. :doc:`methods/similarity-search` explains the ranking. The search returns between 1 and 128 galaxies, 32 by default, and offers:

- **Any position**, set separately for the images and the spectra; otherwise each selected cell or window is compared at the same place;
- for each match, its similarity from 0 to 1, a note when it lacks a selected observation and was matched on AION's prediction, and for each image and spectrum it has, or every one with AION predictions on, a map of where it agrees with the query;
- **Mask**, under which image cells, spectrum windows and catalogue values that score below a threshold lose their colour. Images and spectra each have a threshold for the query and one for the matches, and the catalogue one for the query, each by default the middle of the range of the maps it applies to; **Invert** keeps the colour on what scores below instead.

Click a match's image to search from it: it becomes the selected galaxy, with an empty selection. After a search, the **Mean** map circles whichever of the query galaxy and its matches the morphology filter shows, until another galaxy is selected or another search runs; **Full** circles only the selected galaxy's tokens.

**Whole modes** switches to a :term:`whole-mode search`: a click on an image, a spectrum or a catalogue section (Redshift, Legacy Survey, HSC) selects or deselects that whole mode, outlined, and **Any position** applies to the images and spectra chosen. A mode the galaxy lacks can be chosen when AION predictions show it; the search then reads AION's prediction and says so. Each match shows its score, the mean of the chosen modes' standardised scores, one signed bar per mode for its contribution, and the same-place map of every slot. The cell, window and value selection stays as it was and returns when the switch is turned off; one search never mixes the two.

Before or after a search, the **This galaxy** and **Population** maps shade the query galaxy's own slots by how they relate to the selection; see :doc:`methods/query-maps`.

Each map is coloured on its own scale, from its lowest to its highest value; hover over a cell or a spectral window to read its value.

Search by text
--------------

Describe a galaxy in up to 500 characters. The site returns the 32 image or spectrum matches that lie closest to the description, each with its galaxy, the modality it matched by and its cosine similarity, so a galaxy can appear twice; see :doc:`methods/text-search`.

Download the embeddings
-----------------------

The encoded embeddings, the codebook embeddings and the tokens of every galaxy can be downloaded. :doc:`reference` gives their layout.
