import warnings
from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir

from wordlevel.lexicons.core.load import (
    Lexicon,
    LexiconConnectionError,
    load_lexicon,
    resolve_revision,
    stored_lexicon_paths,
    stored_revision,
    update_lexicons,
)
from wordlevel.lexicons.types.labmt import labMTLanguage
from wordlevel.lexicons.types.socialsent import SocialSentYear


class LexiconLoader:
    """
    An interface for (down)loading pre-defined lexicons that map tokens to scores.

    `wl.lex` is a ready-made loader with the default options, and is all that most uses need.
    Construct a `LexiconLoader` to change where lexicons are cached, which HuggingFace token they
    are downloaded with, or which version of each lexicon is loaded.

    Lexicons are downloaded from
    [HuggingFace](https://huggingface.co/datasets/ryanjgallag/wordlevel-lexicons) and cached
    locally. Each lexicon is versioned separately, so updating one does not change the others.

    Example:
        ```python
        --8<-- "snippets/lexicons/lexicon.py:revision"
        ```

    Args:
        cache_dir:
            The directory that downloaded lexicons are cached in. Defaults to the user's cache
            directory, which is shared by every project on the machine. Pass a directory within a
            project to keep its lexicons separate from other projects.
        hf_token:
            A HuggingFace access token to authenticate downloads with. If not given, the token is
            read from the `HF_TOKEN` environment variable or a local HuggingFace login, if either
            is available.
        revision:
            Which version of each lexicon to load. `"cached"` downloads from HuggingFace the first
            time that a lexicon is requested, and then only accesses that cached lexicon on later
            calls. `"latest"` checks HuggingFace for the latest version on every load,
            updating the cached copy if it has changed; if HuggingFace cannot be reached, the
            cached copy is loaded with a warning. Any other value is parsed as a commit hash,
            branch, or tag of the lexicon repository, and exactly that version is loaded, never a
            substitute. The version a lexicon was loaded at is recorded on its `revision`.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        hf_token: str | None = None,
        revision: Literal["cached", "latest"] | str = "cached",
    ):
        if cache_dir is None:
            cache_dir = user_cache_dir("wordlevel/lexicons")

        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.revision = revision

    def _load(self, lexicon: str, filename: str) -> Lexicon:
        """
        Loads a lexicon at the commit that the loader's `revision` calls for

        Args:
            lexicon:
                The name of the lexicon
            filename:
                The name of the file holding the lexicon's scores within the lexicon's directory
        """
        # The commit that we have stored locally. Can be None
        stored_commit = stored_revision(
            lexicon=lexicon, filename=filename, subfolder=None, cache_dir=self.cache_dir
        )

        commit: str
        if self.revision == "cached":
            # Already cached, so return that commit
            if stored_commit is not None:
                commit = stored_commit
            # Not yet cached, so fetch the latest commit
            else:
                commit = resolve_revision(
                    lexicon=lexicon,
                    filename=filename,
                    subfolder=None,
                    revision="main",
                    hf_token=self.hf_token,
                )
        elif self.revision == "latest":
            try:
                commit = resolve_revision(
                    lexicon=lexicon,
                    filename=filename,
                    subfolder=None,
                    revision="main",
                    hf_token=self.hf_token,
                )
            except LexiconConnectionError:
                if stored_commit is None:
                    raise
                warnings.warn(
                    f"Could not reach HuggingFace to check for a newer version of the '{lexicon}' "
                    f"lexicon, so the cached copy at revision '{stored_commit}' was loaded instead",
                    stacklevel=3,
                )
                commit = stored_commit
        # A specific commit was requested and we already have it
        elif self.revision == stored_commit:
            commit = stored_commit  # type: ignore[assignment]
        # A specific commit was requested and we need to fetch it because we don't have it cached
        else:
            commit = resolve_revision(
                lexicon=lexicon,
                filename=filename,
                subfolder=None,
                revision=self.revision,
                hf_token=self.hf_token,
            )

        return load_lexicon(
            lexicon=lexicon,
            filename=filename,
            revision=commit,
            cache_dir=self.cache_dir,
            hf_token=self.hf_token,
        )

    def update_cached(self) -> None:
        """
        Updates every cached lexicon to its latest version on HuggingFace.

        Only lexicons that have been updated on HuggingFace are downloaded again.

        Example:
            ```python
            --8<-- "snippets/lexicons/lexicon.py:update-cached"
            ```

        Raises:
            LexiconConnectionError: If HuggingFace could not be reached
            LexiconDownloadError: If the lexicons could not be downloaded from HuggingFace
        """
        repo_paths = stored_lexicon_paths(cache_dir=self.cache_dir)
        if not repo_paths:
            return

        update_lexicons(cache_dir=self.cache_dir, repo_paths=repo_paths, hf_token=self.hf_token)

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

        Note:
            The first call for a given language downloads that lexicon from
            [HuggingFace](https://huggingface.co/datasets/ryanjgallag/wordlevel-lexicons) and
            caches it. By default, later calls for the same lexicon read the cached copy without
            contacting HuggingFace. To load the latest or a specific version instead, see
            [`LexiconLoader`][wordlevel.lexicons.lexicon.LexiconLoader].

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
        return self._load(lexicon="labMT", filename=f"{language}.parquet")

    def SocialSent(self, year: SocialSentYear) -> Lexicon:
        """
        A set of English sentiment lexicons constructed for each decade from 1850 to 2000.

        Each lexicon represents the inferred sentiment of the top 5,000 non-stop
        words from the [Corpus of Historical American English](https://www.english-corpora.org/coha/).

        Example:
            ```python
            --8<-- "snippets/lexicons/lexicon.py:socialsent"
            ```

        Note:
            The first call for a given year downloads that lexicon from
            [HuggingFace](https://huggingface.co/datasets/ryanjgallag/wordlevel-lexicons) and
            caches it. By default, later calls for the same lexicon read the cached copy without
            contacting HuggingFace. To load the latest or a specific version instead, see
            [`LexiconLoader`][wordlevel.lexicons.lexicon.LexiconLoader].

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
        return self._load(lexicon="SocialSent", filename=f"{year}.parquet")


lex = LexiconLoader()
