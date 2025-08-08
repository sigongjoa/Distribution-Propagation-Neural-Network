import os
import sys

# Add the project root directory to sys.path to make dpnn_lib discoverable by Sphinx.
# This is crucial for autodoc to find and document modules within dpnn_lib.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DPNN'
copyright = '2025, Your Name'
author = 'Your Name'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically extract documentation from docstrings.
    'sphinx.ext.napoleon', # Support for Google and NumPy style docstrings.
    'sphinx.ext.viewcode', # Add links to highlighted source code.
    'myst_parser',         # Add support for Markdown files.
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'ko'

# Autodoc configuration: Defines default options for how autodoc extracts information.
autodoc_default_options = {
    'members': True,         # Document all public members (functions, classes, methods).
    'undoc-members': True,   # Document members without docstrings.
    'show-inheritance': True,# Show inheritance hierarchy for classes.
}

# Mock imports for modules that might not be available during docs build.
# This prevents Sphinx from failing if these packages are not installed in the build environment.
autodoc_mock_imports = ["torch", "numpy", "scipy", "transformer_components"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']