from wordlevel.api.catalog.catalog import Catalog
from wordlevel.api.comparison.comparison import comp
from wordlevel.api.lexicons.lexicon import lex
from wordlevel.datasets.dataset import Dataset

try:
    from wordlevel._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["__version__", "Catalog", "comp", "Dataset", "lex"]
