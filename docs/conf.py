# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, Path(__file__).parents[1].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "alphaUniverse"
copyright = "2026, Hanbang Wu, Biprateep Dey, Nolan Koblischke, Joshua Speagle"
author = "Hanbang Wu, Biprateep Dey, Nolan Koblischke, Joshua Speagle"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "bokeh.sphinxext.bokeh_plot",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.openapi",
]

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


def schema_block(app, what, name, obj, options, lines):
    if isinstance(obj, pa.Schema):
        lines[:] = ["::", ""] + [f"   {line}" for line in str(obj).splitlines()]


def setup(app):
    app.connect("autodoc-process-docstring", schema_block)
