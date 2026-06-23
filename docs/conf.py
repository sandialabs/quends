import os
import sys

from sphinx_gallery.sorting import ExplicitOrder, _SortKey

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("../src"))  # Adjust this path if necessary
sys.path.insert(0, os.path.abspath("../src/quends"))
sys.path.insert(0, os.path.abspath("../examples/tutorial"))

print("Python path:", sys.path)  # This will help you verify the path

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Local-only authoring notes / migration reports live in docs/ but are not part
# of the published site.  Exclude them so they are not treated as orphan source
# documents (these patterns match nothing in a clean checkout).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*_usage_guide.md",
    "*_cheat_sheet.md",
    "*_migration_report.md",
    "*_alignment_report.md",
    "*_unification_report.md",
    "notebook_tutorial_index.md",
]
master_doc = "index"
project = "QUENDS"
year = "2024"
author = "Bert Debusschere, Pieterjan Robbe, Evans Etrue Howard, Abeyah Calpatura"
copyright = f"{year}, {author}"
try:
    from pkg_resources import get_distribution

    version = release = get_distribution("quends").version
except Exception:
    import traceback

    traceback.print_exc()
    version = release = "0.0.0"

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/sandialabs/quends/issues/%s", "#%s"),
    "pr": ("https://github.com/sandialabs/quends/pull/%s", "PR #%s"),
}

# Documentation theme.  Single switch -- the default is sphinx-immaterial, but a
# maintainer can choose another supported theme without editing this file:
#
#     QUENDS_DOCS_THEME=sphinx_book_theme   sphinx-build -b html docs docs/_build/html
#     QUENDS_DOCS_THEME=pydata_sphinx_theme sphinx-build -b html docs docs/_build/html
#
# (or just change the default string below). The dispatch keeps
# ``html_theme_options`` valid -- and the ``-W`` build clean -- for whichever
# theme is active, and loads the theme's extension only when it needs one.
# The landing page (sphinx-design cards + image badges) and the Sandia colours
# (docs/_static/sandia.css covers both Material's ``--md-*`` variables and the
# pydata/book ``--pst-*`` variables) render under all three themes.
html_theme = os.environ.get("QUENDS_DOCS_THEME", "sphinx_immaterial")

_theme_extensions = []
if html_theme == "sphinx_immaterial":
    _theme_extensions = ["sphinx_immaterial"]
    html_theme_options = {
        "site_url": "https://sandialabs.github.io/quends/",
        "repo_url": "https://github.com/sandialabs/quends/",
        "repo_name": "quends",
        "icon": {"repo": "fontawesome/brands/github"},
        "features": [
            "navigation.tabs",
            "navigation.tabs.sticky",
            "navigation.top",
            "navigation.sections",
            "toc.follow",
            "search.highlight",
            "search.share",
            "content.code.copy",
        ],
        "palette": [
            {
                "scheme": "default",
                "primary": "cyan",
                "accent": "light-blue",
                "toggle": {
                    "icon": "material/weather-night",
                    "name": "Switch to dark mode",
                },
            },
            {
                "scheme": "slate",
                "primary": "cyan",
                "accent": "light-blue",
                "toggle": {
                    "icon": "material/weather-sunny",
                    "name": "Switch to light mode",
                },
            },
        ],
    }
elif html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "github_url": "https://github.com/sandialabs/quends",
        "use_edit_page_button": False,
        "navigation_with_keys": False,
    }
elif html_theme == "sphinx_book_theme":
    html_theme_options = {
        "repository_url": "https://github.com/sandialabs/quends",
        "use_repository_button": True,
        "use_issues_button": True,
        "home_page_in_toc": False,
    }
else:
    html_theme_options = {}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_short_title = f"{project}-{version}"

# Sandia National Laboratories brand colors (override the Material palette).
html_static_path = ["_static"]
html_css_files = ["sandia.css"]

napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_use_param = False

autodoc_docstring_signature = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Append (NOT reassign) so the base extensions above -- notably
# sphinx.ext.napoleon, which converts the NumPy/Google docstring sections --
# are preserved.  Reassigning here previously dropped napoleon and produced
# dozens of malformed-docstring warnings.
extensions += [
    "myst_parser",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx_automodapi.automodapi",
    "sphinx_gallery.gen_gallery",
]

# Load the active theme's extension (only sphinx-immaterial needs one).
extensions += _theme_extensions


examples_tutorial = [
    "datastream_guide.py",
    "ensemble_guide.py",
    "robustworkflow_guide.py",
    "robustworkflow_advanced_guide.py",
]


class ExamplesExplicitOrder(_SortKey):

    def __call__(self, filename):
        # Listed files keep the explicit order above; anything else sorts after.
        try:
            return examples_tutorial.index(filename)
        except ValueError:
            return len(examples_tutorial)


sphinx_gallery_conf = {
    "examples_dirs": "../examples/tutorial",
    "gallery_dirs": "auto_tutorials",
    "subsection_order": ExplicitOrder(
        [
            "../examples/tutorial",
        ]
    ),
    "within_subsection_order": ExamplesExplicitOrder,
    "filename_pattern": r".*",
    "matplotlib_animations": True,
    "image_scrapers": ("matplotlib",),
}

extensions += ["autoapi.extension"]

autoapi_dirs = ["../src/quends"]  # Where the QUENDS source code is
autoapi_type = "python"
autoapi_add_toctree_entry = True

autoapi_template_dir = "_templates/autoapi"  # Templates for AutoAPI documentation
suppress_warnings = ["autoapi", "config.cache"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_typehints = "description"
autoapi_own_page_level = "module"
autoapi_keep_files = True  # Keep the AutoAPI generated files on the filesystem

print("AutoAPI directories:", autoapi_dirs)
