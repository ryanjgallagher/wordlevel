from pathlib import Path

from platformdirs import user_cache_dir

from wordlevel.api.lexicons.types.labmt import labMTLanguage
from wordlevel.api.lexicons.types.socialsent import SocialSentYear
from wordlevel.core.lexicons.load import Lexicon, load_lexicon


class LexiconLoader:
    """
    An interface for (down)loading pre-defined lexicons that map tokens to scores.

    Lexicons are downloaded the first time they are loaded and cached locally. Later loads of the
    same lexicon read from the local cache.

    Args:
        cache_dir:
            The directory that downloaded lexicons are cached in. Defaults to the user's cache
            directory.
        hf_token:
            A HuggingFace access token to authenticate downloads with. If not given, the token is
            read from the `HF_TOKEN` environment variable or a local HuggingFace login, if either
            is available.
    """

    def __init__(self, cache_dir: str | Path | None = None, hf_token: str | None = None):
        if cache_dir is None:
            cache_dir = user_cache_dir("wordlevel/lexicons")

        self.cache_dir = cache_dir
        self.hf_token = hf_token

    def labMT(self, language: labMTLanguage) -> Lexicon:
        """
        A set of "happiness" sentiment lexicons constructed for various languages.

        The 10,000 most frequent words for each language were rated by Mechanical Turk (MT) workers
        on a scale from 1 to 9, where 1 is the most negative ("saddest") and 9 is the most positive
        ("happiest"). The words were sourced from several sources, including the Google Books
        project, the Google Web Crawl, Twitter, the New York Times, music lyrics, and movie and
        television titles.

        Example:
            ```python
            --8<-- "snippets/lexicons/lexicon.py:labmt"
            ```

        License:
            This data is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
            International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

        Citation:
            If you use the English lexicon, please cite the following paper.

            > Dodds, P. S., Harris, K. D., Kloumann, I. M., Bliss, C. A., & Danforth, C. M. (2011).
            [Temporal patterns of happiness and information in a global social network:
            Hedonometrics and
            Twitter](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0026752).
            PLoS ONE, 6(12), e26752.

            If you use any of the non-English lexicons, please cite the following paper.

            > Dodds, P. S., Clark, E. M., Desu, S., Frank, M. R., Reagan, A. J., Williams, J. R.,
            Mitchell, L., Harris, K. D., Kloumann, I. M., Bagrow, J. P., Megerdoomian, K., McMahon,
            M. T., Tivnan, B. F., & Danforth, C. M. (2015). [Human language reveals a universal
            positivity bias](https://www.pnas.org/doi/10.1073/pnas.1411678112). Proceedings of the
            National Academy of Sciences, 112(8), 2389-2394.

        Args:
            language: The language of the lexicon

        Raises:
            LexiconDownloadError: If the lexicon could not be downloaded from HuggingFace
        """
        return load_lexicon(
            lexicon="labMT",
            filename=f"{language}.parquet",
            cache_dir=self.cache_dir,
            hf_token=self.hf_token,
        )

    def SocialSent(self, year: SocialSentYear) -> Lexicon:
        """
        A set of English sentiment lexicons constructed for each decade from 1850 to 2000.

        Each lexicon represents the inferred sentiment of the top 5,000 non-stop
        words from the [Corpus of Historical American English](https://www.english-corpora.org/coha/).

        Example:
            ```python
            --8<-- "snippets/lexicons/lexicon.py:socialsent"
            ```

        License:
            This data is made available under the [Public Domain Dedication and License
            v1.0](https://opendatacommons.org/licenses/pddl/1-0/).

        Citation:
            If you use this lexicon, please cite the following paper.

            > Hamilton, W. L., Clark, K., Leskovec, J., & Jurafsky, D. (2016). [Inducing
            domain-specific sentiment lexicons from unlabeled
            corpora](https://aclanthology.org/D16-1057/). In Proceedings of the 2016 Conference on
            Empirical Methods in Natural Language Processing (pp. 595-605).

        Args:
            year: The year of the lexicon

        Raises:
            LexiconDownloadError: If the lexicon could not be downloaded from HuggingFace
        """
        return load_lexicon(
            lexicon="SocialSent",
            filename=f"{year}.parquet",
            cache_dir=self.cache_dir,
            hf_token=self.hf_token,
        )


lex = LexiconLoader()
