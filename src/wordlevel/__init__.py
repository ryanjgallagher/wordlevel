from wordlevel.api.catalog.catalog import Catalog
from wordlevel.comparison.comparison import comp
from wordlevel.datasets.dataset import Dataset
from wordlevel.lexicons.lexicon import LexiconLoader, lex


try:
    from wordlevel._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["Catalog", "Dataset", "LexiconLoader", "__version__", "comp", "lex"]
