# Install

`wordlevel` runs on Python 3.10 and above. It can be installed with any Python package manager,
including [pip] and [uv]. If you want to use uv, make sure to [install it first][uv install].

  [pip]: https://pip.pypa.io/
  [uv]: https://docs.astral.sh/uv/
  [uv install]: https://docs.astral.sh/uv/getting-started/installation/

## Create a virtual environment

Creating a virtual environment is optional, but it keeps `wordlevel` and its dependencies separate
from other projects on your computer. From the directory where you want to work, run the following
commands:

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

=== "uv"

    ```bash
    uv venv
    source .venv/bin/activate
    ```

## Install `wordlevel`

To install `wordlevel`, simply run one of the following commands:

=== "pip"

    ```bash
    pip install wordlevel
    ```

=== "uv"

    ```bash
    uv pip install wordlevel
    ```

## Saving plots

`wordlevel` uses [Altair] to generate plots. To save plots as PNGs, SVGs, or PDFs, some extra
dependencies are required. They are bundled with the `save` extra when installing `wordlevel`:

=== "pip"

    ```bash
    pip install "wordlevel[save]"
    ```

=== "uv"

    ```bash
    uv pip install "wordlevel[save]"
    ```

  [Altair]: https://altair-viz.github.io/

## Check that it worked

If the following command prints a version number, you are ready to go!

```bash
python -c "import wordlevel; print(wordlevel.__version__)"
```
