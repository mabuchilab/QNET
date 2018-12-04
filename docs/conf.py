# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys
import datetime


import qnet

from unittest import mock
MOCK_MODULES = ['numpy', 'numpy.linalg', 'scipy', 'scipy.sparse', 'matplotlib',
    'matplotlib.pyplot', 'qutip', 'ply','pyx','pyx.text']
sys.modules.update((mod_name, mock.Mock()) for mod_name in MOCK_MODULES)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('_extensions'))

# -- Generate API documentation ------------------------------------------------
def run_apidoc(_):
    """Generage API documentation"""
    import better_apidoc
    better_apidoc.main([
        'better-apidoc',
        '-t',
        os.path.join('.', '_templates'),
        '--force',
        '--no-toc',
        '--separate',
        '-o',
        os.path.join('.', 'API'),
        os.path.join('..', 'src', 'qnet'),
    ])


# -- General configuration -----------------------------------------------------

# Report broken links as warnings
nitpicky = True

extensions = [
    'graphviz_ext',
    'inheritance_diagram',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
    'dollarmath',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6', None),
    'sympy': ('http://docs.sympy.org/latest/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'qutip': ('http://qutip.org/docs/latest/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'
project = 'QNET'
year = str(datetime.datetime.now().year)
author = 'Nikolas Tezak and Michael Goerz'
copyright = u'2012-2018, Nikolas Tezak, Michael Goerz'
version = release = qnet.__version__

pygments_style = 'friendly'
extlinks = {
    'issue': ('https://github.com/mabuchilab/QNET/issues/%s', '#'),
    'pr': ('https://github.com/mabuchilab/QNET/pull/%s', 'PR #'),
}

# autodoc settings
autoclass_content = 'both'
autodoc_member_order = 'bysource'


html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

# Mathjax settings
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js'
mathjax_config = {
    'extensions': [
        'tex2jax.js', 'AMSmath.js', 'AMSsymbols.js', 'noErrors.js',
        'noUndefined.js'],
    'jax': ['input/TeX', 'output/SVG'],
    'TeX': {
        'Macros': {
            'tr': ['{\\operatorname{tr}}', 0],
            'Tr': ['{\\operatorname{tr}}', 0],
            'diag': ['{\\operatorname{diag}}', 0],
            'abs': ['{\\operatorname{abs}}', 0],
            'pop': ['{\\operatorname{pop}}', 0],
            'SLH': ['{\\operatorname{SLH}}', 0],
            'aux': ['{\\text{aux}}', 0],
            'opt': ['{\\text{opt}}', 0],
            'tgt': ['{\\text{tgt}}', 0],
            'init': ['{\\text{init}}', 0],
            'lab': ['{\\text{lab}}', 0],
            'rwa': ['{\\text{rwa}}', 0],
            'fwhm': ['{\\text{fwhm}}', 0],
            'bra': ['{\\langle#1\\vert}', 1],
            'ket': ['{\\vert#1\\rangle}', 1],
            'Bra': ['{\\left\\langle#1\\right\\vert}', 1],
            'Braket': ['{\\left\\langle #1\\vphantom{#2} \\mid #2\\vphantom{#1}\\right\\rangle}', 2],
            'Ket': ['{\\left\\vert#1\\right\\rangle}', 1],
            'mat': ['{\\mathbf{#1}}', 1],
            'op': ['{\\hat{#1}}', 1],
            'Op': ['{\\hat{#1}}', 1],
            'dd': ['{\\,\\text{d}}', 0],
            'daggered': ['{^{\\dagger}}', 0],
            'transposed': ['{^{\\text{T}}}', 0],
            'Liouville': ['{\\mathcal{L}}', 0],
            'DynMap': ['{\\mathcal{E}}', 0],
            'identity': ['{\\mathbf{1}}', 0],
            'Norm': ['{\\lVert#1\\rVert}', 1],
            'Abs': ['{\\left\\vert#1\\right\\vert}', 1],
            'avg': ['{\\langle#1\\rangle}', 1],
            'Avg': ['{\\left\langle#1\\right\\rangle}', 1],
            'AbsSq': ['{\\left\\vert#1\\right\\vert^2}', 1],
            'Re': ['{\\operatorname{Re}}', 0],
            'Im': ['{\\operatorname{Im}}', 0],
        }
    }
}


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------

from sphinx.ext.napoleon.docstring import GoogleDocstring

# first, we define new methods for any new sections and add them to the class
def parse_keys_section(self, section):
    return self._format_fields('Keys', self._consume_fields())
GoogleDocstring._parse_keys_section = parse_keys_section

def parse_attributes_section(self, section):
    return self._format_fields('Attributes', self._consume_fields())
GoogleDocstring._parse_attributes_section = parse_attributes_section

def parse_class_attributes_section(self, section):
    return self._format_fields('Class Attributes', self._consume_fields())
GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


# -- Monkeypatch for instance attribs (sphinx bug #2044) -----------------------


from sphinx.ext.autodoc import (
    ClassLevelDocumenter, InstanceAttributeDocumenter)


def iad_add_directive_header(self, sig):
    ClassLevelDocumenter.add_directive_header(self, sig)


InstanceAttributeDocumenter.add_directive_header = iad_add_directive_header


# -- Documenter for Singletons -------------------------------------------------

from sphinx.ext.autodoc import DataDocumenter


class SingletonDocumenter(DataDocumenter):
    directivetype = 'data'
    objtype = 'singleton'
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, qnet.utils.singleton.SingletonType)


# -- Options for HTML output ---------------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_theme = 'sphinxdoc'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'collapse_navigation': True,
    'display_version': True,
}

inheritance_graph_attrs = dict(size='""')
graphviz_output_format = 'svg'

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = 'favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# -----------------------------------------------------------------------------
def setup(app):
    app.add_autodocumenter(SingletonDocumenter)
    app.connect('builder-inited', run_apidoc)
