from __future__ import annotations

from datetime import datetime

project = "IntronModel"
author = "IntronModel contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown"}
root_doc = "index"

html_theme = "alabaster"
html_title = "IntronModel Documentation"
html_static_path = ["_static"]
