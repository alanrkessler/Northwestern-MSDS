# Atom Setup

-   Add the following packages to Atom:
    -   linter
    -   linter-markdown
    -   file-icons
    -   Hydrogen
    -   platformio-ide-terminal

## Python

-   Install Anaconda3. On Windows, make sure that Anaconda is selected as the default Python and that Anaconda is added to the PATH environment variable
-   Use `conda install nb_conda flake8 pylint pydocstyle` from the terminal or Anaconda Prompt on Windows to install necessary packages in the base environment
-   Add the following linters to Atom:
    -   linter-flake8
    -   linter-pylint
    -   linter-pydocstyle
-   For reference, assume that base conda environment is where linters will run
-   For a new conda environment, run `python -m ipykernel install --user --name "py36"`
-   On Windows, the "platformio-ide-terminal" can refer to PowerShell by default. Other options include:
    -   Ubuntu: Navigate to "Use developer features" and enable "Developer mode". Then navigate to "Turn Windows features on and off". Turn check the box for "Windows Subsystem for Linux". After restarting, install Ubuntu from the Windows Store. Run Ubuntu to configure. In the "platformio-ide-terminal" settings, change the "Shell Override" to  `ubuntu.exe`.
    -   Anaconda Prompt: In the "platformio-ide-terminal" settings, change the "Shell Override" to  `C:\Windows\System32\cmd.exe`. Set the "Auto Run Command" to: `<conda install location>\Anaconda3\Scripts\activate.bat`. If Python is not installed in the Windows Subsystem for Linux, this option is preferred for managing conda environments and packages.

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
