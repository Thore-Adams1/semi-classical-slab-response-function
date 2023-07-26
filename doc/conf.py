# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Semi Classical Slab Response Function'
copyright = '2023, Sam Praill, Charlotte Lawton, Hasan Balable, Hai-Yao Deng'
author = 'Sam Praill, Charlotte Lawton, Hasan Balable, Hai-Yao Deng'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Build automatic api docs
    'sphinx.ext.autodoc', 
    # Interpret google-style docstrings
    'sphinx.ext.napoleon',
    # Allow for testing of code snippets in documentation
    'sphinx.ext.doctest',
    # Allow documentation of cli scripts
    'sphinxarg.ext'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_type_aliases = {
    'ArrayLike': 'ArrayLike'
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
