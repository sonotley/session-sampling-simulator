# session-sampling-simulator

An application that generates synthetic database sessions and allows you to simulate the effects of sampling those sessions to understand the error characteristics.

> **Warning**
> I'm still getting this repo set up and tidying the code, so the builds referred to below don't actually exist yet.

## Installing

The application is distributed as a Python package. 
The recommended way to use it is by installing from PyPI using either pipx or uv.

    pipx install session_sampling_simulator

or

    uv tool install session_sampling_simulator

Equivalently you can download the wheel or sdist from the GitHub releases page and substitute the path to the file for `session_sampling_simulator` in either command.

Of course, you can also just create a virtual environment manually, activate it and install the package.

    python -m venv .venv
    source .venv/bin/activate
    pip install session_sampling_simulator

## Using

The package installs a CLI script `sesasim` which you can invoke to access various functionality. 
You can start the GUI mode with `sesasim gui <mode>`.
