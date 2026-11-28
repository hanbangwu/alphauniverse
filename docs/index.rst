alphaUniverse
=============

alphaUniverse is a Python library for searching the galaxies of the COSMOS field through AION :cite:p:`parker2025aion`, Polymathic AI's foundation model for astronomy: by parts of a galaxy's image, spectrum or catalog values, by whole surveys, or in plain English, and reading what AION expects each galaxy to look like (:doc:`library/index`).

The alphaUniverse website is built on the library. On it you can:

- **browse the sky** with every galaxy marked on the Legacy Survey image;
- **browse a UMAP** of every galaxy, where galaxies AION finds alike sit together, colored by redshift, color, size or spectral features;
- **see what AION expects** a galaxy to look like in every survey, including surveys that never observed it;
- **find similar galaxies** from the parts of one galaxy's image, spectrum or catalog values you pick;
- **search in plain English**, for example "a quiescent galaxy with no ongoing star formation".

:doc:`using` shows how. :doc:`concepts` explains what AION does, assuming no machine-learning background.

The data
--------

``hanbangwu/alphauniverse-cosmos`` is Legacy Survey DR10 south over a √2° × √2° box on COSMOS, crossmatched against HSC PDR3, DESI EDR SV3 and SDSS. AION reads all four:

================== ======== =====================================================
Survey             Modality What AION reads
================== ======== =====================================================
Legacy Survey DR10 image    96 × 96 px (25″) *g r i z* cutout; 12 catalog values
HSC PDR3           image    96 × 96 px (16″) *g r i z y* cutout; 13 catalog values
DESI EDR SV3       spectrum 3500–10463 Å, resampled to 0.8 Å
SDSS               spectrum 3500–10463 Å, resampled to 0.8 Å
================== ======== =====================================================

It also reads one spectroscopic redshift per galaxy: DESI's if unflagged and at most 6, else SDSS's under the same test.

.. toctree::
   :caption: Library

   library/index
   library/reference

.. toctree::
   :caption: Website

   using
   concepts
   methods/map
   methods/similarity-search
   methods/query-maps
   methods/text-search
   glossary
   references

.. toctree::
   :caption: Developers

   dev/how-to
   dev/reference
   dev/behavior
   dev/benchmarks
