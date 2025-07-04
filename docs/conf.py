import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("../src"))  # Adjust this path if necessary
sys.path.insert(0, os.path.abspath("../src/quends"))

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
source_suffix = ".rst"
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

extensions += ["autoapi.extension"]

autoapi_dirs = ["../src/quends"]  # Where the QUENDS source code is
autoapi_type = "python"
autoapi_add_toctree_entry = True

autoapi_template_dir = "_templates/autoapi"  # Templates for AutoAPI documentation
suppress_warnings = ["autoapi"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_own_page_level = "module"
autoapi_keep_files = False  # Keep the AutoAPI generated files on the filesystem

html_theme = "sphinx_book_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
