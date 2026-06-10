# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "iguanas"
copyright = "Mozilla Public License (MPL) 2.0"
author = "Charles Poli"
release = "1.0.x"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- SEO Configuration -------------------------------------------------------

# Base URL for sitemap and canonical links
html_baseurl = "https://paypal.github.io/Iguanas/"

# SEO-optimized title and description
html_title = "Iguanas - Lightning Fast Rule generation with Polars"
html_short_title = "Iguanas"

# Sitemap configuration
sitemap_url_scheme = "{link}"

# OpenGraph configuration for social media
ogp_site_url = "https://paypal.github.io/Iguanas/"
ogp_image = "https://paypal.github.io/Iguanas/_static/IGUANAS_LOGO.png"
ogp_description_length = 160
ogp_type = "website"
ogp_site_name = "Iguanas - Lightning Fast Rule generation with Polars"
ogp_custom_meta_tags = [
    '<meta name="description" content="High-performance rule generation library with sklearn-compatible API." />',
    '<meta name="keywords" content="data science, machine learning, rule generation, python, polars, sklearn" />',
]

# Include verification and robots files
html_extra_path = [".nojekyll"]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Allow errors in notebooks

# Suppress warnings
suppress_warnings = [
    "ref.python",  # Suppress ambiguous cross-reference warnings for Python objects
    "autodoc.duplicate",  # Suppress duplicate object description warnings
    "sphinx_autodoc_typehints.forward_reference",  # Suppress Pydantic forward reference warnings
]

# Napoleon settings for NumPy docstring style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = False

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__init__, __new__, __weakref__, __dict__, __module__, __len__, __getitem__, __setitem__, __delitem__, __iter__, __next__, __repr__, __str__",
}
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "fully-qualified"
autodoc_type_aliases = {
    "polars.series.series.Series": "polars.Series",
    "polars.dataframe.frame.DataFrame": "polars.DataFrame",
}
autodoc_class_signature = "separated"

html_favicon = "_static/iguanas.ico"
html_logo = "_static/IGUANAS_LOGO.png"
html_css_files = ["iguanas.css"]


# Custom filter to exclude Pydantic fields and validators
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip Pydantic fields, validators, and internal members."""
    import inspect

    from pydantic.fields import FieldInfo, ModelPrivateAttr

    # Allow _Base* classes (base classes for documentation)
    if name.startswith("_Base"):
        return None

    # Skip other private members
    if name.startswith("_"):
        return True

    # Skip Pydantic model methods
    if name.startswith("model_"):
        return True

    # Skip validator methods
    if name.startswith(("check_", "validate_")):
        return True

    # Skip if it's an attribute/data descriptor/property
    if what in ("attribute", "data", "property"):
        return True

    # Skip if the object is a FieldInfo (Pydantic field descriptor)
    if isinstance(obj, FieldInfo | ModelPrivateAttr):
        return True

    # Check if it's a descriptor or data attribute (not a method)
    if not (inspect.ismethod(obj) or inspect.isfunction(obj) or callable(obj)):
        # It's likely a data attribute
        return True

    return None


def remove_attributes_section(app, what, name, obj, options, lines):
    """Remove Attributes section from docstrings and strip type-role markup."""
    # strip type role markup like :py:class:`str` or :py:data:`list`
    pattern = re.compile(r":py:(?:class|data|obj|func|meth|attr|mod):`~?([^`]+)`")
    for idx, line in enumerate(lines):
        lines[idx] = pattern.sub(r"\1", line)

    if what in ("class", "exception"):
        i = 0
        while i < len(lines):
            line = lines[i].strip() if i < len(lines) else ""

            # Check if this is an Attributes section header
            if line == "Attributes":
                # Check if next line is the underline (-----)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and all(c == "-" for c in next_line):
                        # Found Attributes section, now find where it ends
                        start = i
                        i += 2  # Skip "Attributes" and "----------"

                        # Continue until we hit another section header (word followed by dashes)
                        # or until the end of the docstring
                        while i < len(lines):
                            current = lines[i].strip() if i < len(lines) else ""
                            # Look ahead to see if next line is a dash underline (section header)
                            if current and i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if next_line and all(c == "-" for c in next_line):
                                    # Found next section, stop here
                                    break
                            i += 1

                        # Remove the Attributes section (from start to current position)
                        del lines[start:i]
                        # Don't increment i since we deleted lines
                        continue
            i += 1


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    """Replace long Polars type names and remove type-role markup in autodoc signatures."""

    def strip_type_roles(text: str) -> str:
        return re.sub(
            r":py:(?:class|data|obj|func|meth|attr|mod):`~?([^`]+)`",
            r"\1",
            text,
        )

    if signature is not None:
        signature = signature.replace("polars.series.series.Series", "polars.Series").replace(
            "polars.dataframe.frame.DataFrame", "polars.DataFrame"
        )
        signature = strip_type_roles(signature)
    if return_annotation is not None:
        return_annotation = return_annotation.replace(
            "polars.series.series.Series", "polars.Series"
        ).replace("polars.dataframe.frame.DataFrame", "polars.DataFrame")
        return_annotation = strip_type_roles(return_annotation)
    return signature, return_annotation


def remove_builtin_type_links(app, doctree, docname):
    """Convert builtin Python type references into plain text."""
    from docutils import nodes

    builtin_types = {
        "str",
        "list",
        "dict",
        "tuple",
        "set",
        "float",
        "int",
        "bool",
        "None",
        "NoneType",
    }

    for node in doctree.traverse(nodes.reference):
        refuri = node.get("refuri", "")
        if refuri.startswith("https://docs.python.org/3") and node.astext() in builtin_types:
            node.replace_self(nodes.Text(node.astext()))


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("autodoc-process-docstring", remove_attributes_section)
    app.connect("autodoc-process-signature", autodoc_process_signature)
    app.connect("doctree-resolved", remove_builtin_type_links)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_toc_level": 2,
    "navbar_align": "content",
    "navigation_depth": 4,
    "show_nav_level": 1,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    # Repository integration
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/paypal/Iguanas",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/iguanas/",
            "icon": "fa-solid fa-box",
        },
    ],
    "use_edit_page_button": True,
    # Google Analytics
    "analytics": {
        "google_analytics_id": "G-Z5TR8D9VTQ",
    },
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

# GitHub context for edit button
html_context = {
    "github_user": "paypal",
    "github_repo": "Iguanas",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
