Using the site
==============

This page says what you can do on the site and what each result means. The methods behind each feature are in the Methods pages.

Browse the map
--------------

The map shows every galaxy, or every token, placed by :doc:`methods/map`. Switch between **Mean**, one point per galaxy, and **Full**, one point per token embedding of every galaxy. Points are coloured by Galaxy Zoo 10 morphology; most galaxies have no label. Filter by morphology to show only some classes. Hover over a point to see the galaxy's image and label, and click it to select the galaxy; in Full, any of a galaxy's tokens selects it.

Look at a galaxy
----------------

A selected galaxy shows, in groups that can each be collapsed:

- its Legacy Survey and HSC images, the central 96 × 96 px of each: the Legacy Survey colour image, and the HSC image coloured from its *g r i z* flux by the mapping that made the Legacy Survey's;
- its DESI and SDSS spectra, smoothed for display with a Gaussian of σ = 2 pixels; search uses AION's predictions, never the smoothed flux;
- its catalogue values, with Galaxy Zoo 10, PROVABGS and the redshift; the 26 values AION reads can be selected for search;
- which surveys it is matched in.

Search by similarity
--------------------

Start a search from the selected galaxy and pick what to compare: cells of its Legacy Survey or HSC image, windows of its DESI or SDSS spectrum, catalogue values, or any mix. :doc:`methods/similarity-search` explains the ranking. The search returns between 1 and 128 galaxies, 32 by default, and offers:

- **Any position**, set separately for the images and the spectra; otherwise each selected cell or window is compared at the same place;
- for each match, its similarity from 0 to 1, a note when it lacks a selected observation and was matched on AION's prediction, and for each image and spectrum it has, a map of where it agrees with the query;
- **Mask**, which adds a panel marking the image cells at or above a threshold, by default the middle of the panel's range; **Invert** marks those below.

Before or after a search, the **This galaxy** and **Population** maps shade the query galaxy's own slots by how they relate to the selection; see :doc:`methods/query-maps`.

Each map is coloured on its own scale, from its lowest to its highest value; hover over a cell or a spectral window to read its value.

Search by text
--------------

Describe a galaxy in up to 500 characters. The site returns the 32 image or spectrum matches that lie closest to the description, each with its galaxy, the modality it matched by and its cosine similarity, so a galaxy can appear twice; see :doc:`methods/text-search`.

Download the embeddings
-----------------------

The encoded embeddings, the codebook embeddings and the tokens of every galaxy can be downloaded. :doc:`reference` gives their layout.
