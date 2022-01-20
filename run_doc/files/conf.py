import os
import sys
import iguanas

sys.path.insert(0, '..')
project = 'Iguanas'
copyright = '2021, James Laidler'
author = 'James Laidler'
version = iguanas.__version__
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.autosummary',
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    'undoc-members': True,
    'member-order': 'bysource',
    "inherited-members": True
}
autodoc_typehints = 'signature'
templates_path = ['_templates']
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
todo_include_todos = True
master_doc = 'index'
numpydoc_show_class_members = False
autosummary_generate = True
panels_add_bootstrap_css = False
html_use_index = False
html_domain_indices = False
html_theme = 'pydata_sphinx_theme'
html_css_files = ['css/iguanas.css']
html_static_path = ['_static']
html_logo = '_static/iguanas_text_logo.png'
html_favicon = '_static/iguanas_icon.png'
html_theme_options = {
    "logo_link": "index",
    "github_url": "https://github.com/paypal/Iguanas",
    "show_prev_next": False
}
