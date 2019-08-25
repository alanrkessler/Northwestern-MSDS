# Atom Setup

-   Add the following packages to Atom:
    -   linter
    -   linter-markdown
    -   file-icons
    -   Hydrogen

## Python

-   Install Anaconda3. On Windows, make sure that Anaconda is selected as the default Python and that Anaconda is added to the PATH environment variable
-   Use `conda install nb_conda flake8 pylint pydocstyle` from the terminal or Anaconda Prompt on Windows to install necessary packages in the base environment
-   Add the following linters to Atom:
    -   linter-flake8
    -   linter-pylint
    -   linter-pydocstyle
-   Change directory to start kernel in to "Current directory of the file" just out of personal preference in using notebooks.
-   For reference, assume that base conda environment is where linters will run
-   For a new conda environment, run `python -m ipykernel install --user --name "py36"`

## R

-   Install the latest version of R
-   Uninstall previous versions of R if no longer needed
-   Update or install R Studio
-   On MacOS, Xcode is required so run `xcode-select --install` from the terminal
-   Install "lintr" and "IRkernel" packages for R
-   Add the "atom-language-r" package to Atom
-   Add the "linter-lintr" package to Atom but do not add the langue dependency which is an older unmaintained equivalent of "atom-language-r"
-   Start R from the terminal or Command Prompt and run `IRkernel::installspec()` because it will not work from R Studio
-   Restart Atom
