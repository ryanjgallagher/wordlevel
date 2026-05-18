from platformdirs import user_cache_dir

from wordlevel.api.lexicons.types.labmt import labMTLanguage
from wordlevel.api.lexicons.types.socialsent import SocialSentYear
from wordlevel.core.lexicons.load import Lexicon, load_lexicon


class LexiconLoader:
    def __init__(self, cache_dir: str | None = None):
        """ """
        if cache_dir is None:
            cache_dir = user_cache_dir("wordlevel/lexicons")

        self.cache_dir = cache_dir

    def labMT(self, language: labMTLanguage) -> Lexicon:
        """ """
        return load_lexicon(
            lexicon="labMT", filename=f"{language}.parquet", cache_dir=self.cache_dir
        )

    def SocialSent(self, year: SocialSentYear) -> Lexicon:
        """ """
        return load_lexicon(
            lexicon="SocialSent", filename=f"{year}.parquet", cache_dir=self.cache_dir
        )


lex = LexiconLoader()
