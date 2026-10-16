alphaUniverse
=============

alphaUniverse shows how AION :cite:p:`parker2025aion`, Polymathic AI's foundation model for astronomy, sees the galaxies of the COSMOS field. You can browse every galaxy on a map of AION's internal representation, ask AION which galaxies resemble one you pick in the image regions, wavelengths or catalogue values you choose, and search the field in plain English.

The data
--------

``hanbangwu/alphauniverse-cosmos`` is Legacy Survey DR10 south over a √2° × √2° box on COSMOS, crossmatched against HSC PDR3, DESI EDR SV3, SDSS, Galaxy Zoo 10 and PROVABGS. AION reads four of these surveys:

================== ======== ===============================================
Survey             Modality What AION reads
================== ======== ===============================================
Legacy Survey DR10 image    96 × 96 px *g r i z* cutout; 12 catalogue values
HSC PDR3           image    96 × 96 px *g r i z y* cutout; 13 catalogue values
DESI EDR SV3       spectrum 3500–10463 Å, resampled to 0.8 Å
SDSS               spectrum 3500–10463 Å, resampled to 0.8 Å
================== ======== ===============================================

It also reads one spectroscopic redshift per galaxy: DESI's if unflagged and at most 6, else SDSS's under the same test. Galaxy Zoo 10 morphologies colour the map; PROVABGS stellar masses, star formation rates and redshifts are the ground truth in the text search and probe benchmarks, and its masses and star formation rates are the labels every galaxy's predicted ones are fitted to.

.. toctree::
   :caption: Background

   concepts
   glossary

.. toctree::
   :caption: Methods

   methods/map
   methods/similarity-search
   methods/query-maps
   methods/text-search

.. toctree::
   :caption: Using the site

   using

.. toctree::
   :caption: Project

   benchmarks
   how-to
   reference
   references
