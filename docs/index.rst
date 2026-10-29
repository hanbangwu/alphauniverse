alphaUniverse
=============

alphaUniverse is a website for exploring the galaxies of the COSMOS field through AION :cite:p:`parker2025aion`, Polymathic AI's foundation model for astronomy. On it you can:

- **browse a map** of every galaxy, where galaxies AION finds alike sit together, coloured by redshift, stellar mass or star formation;
- **see what AION expects** a galaxy to look like in every survey, including surveys that never observed it;
- **find similar galaxies** from the parts of one galaxy's image, spectrum or catalogue values you pick;
- **search in plain English**, for example "a quiescent galaxy with no ongoing star formation".

:doc:`using` shows how. :doc:`concepts` explains what AION does, assuming no machine-learning background.

The data
--------

``hanbangwu/alphauniverse-cosmos`` is Legacy Survey DR10 south over a √2° × √2° box on COSMOS, crossmatched against HSC PDR3, DESI EDR SV3, SDSS, Galaxy Zoo 10 and PROVABGS. AION reads four of these surveys:

================== ======== =====================================================
Survey             Modality What AION reads
================== ======== =====================================================
Legacy Survey DR10 image    96 × 96 px (25″) *g r i z* cutout; 12 catalogue values
HSC PDR3           image    96 × 96 px (16″) *g r i z y* cutout; 13 catalogue values
DESI EDR SV3       spectrum 3500–10463 Å, resampled to 0.8 Å
SDSS               spectrum 3500–10463 Å, resampled to 0.8 Å
================== ======== =====================================================

It also reads one spectroscopic redshift per galaxy: DESI's if unflagged and at most 6, else SDSS's under the same test. Galaxy Zoo 10 labels are shown with each galaxy's catalogue values. :term:`PROVABGS` stellar masses and star formation rates are the measured values the predicted ones are fitted to.

.. toctree::
   :caption: Guide

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
   dev/behaviour
   dev/benchmarks
