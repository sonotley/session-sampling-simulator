# session-sampling-simulator

An application that generates synthetic database sessions and allows you to simulate the effects of sampling those sessions to understand the error characteristics.

## Installing

The application is distributed as a Python package. 
The recommended way to use it is by installing from PyPI using either pipx or uv.

    pipx install session_sampling_simulator

or

    uv tool install session_sampling_simulator

Equivalently you can download the wheel or sdist from the GitHub releases page and substitute the path to the file for `session_sampling_simulator` in either command.

### Warning for uv users!

The standalone Python versions that uv uses don't place nicely with tkinter and matplotlib.
I've implemented a workaround for the GUI, but GUI and charts together are a no-no right now
which means the `gui analyzer` command is broken. 
I'm working on a 'compatibility mode' that will allow this to work.

Of course, you can also just create a virtual environment manually, activate it and install the package.

    python -m venv .venv
    source .venv/bin/activate
    pip install session_sampling_simulator

## Using

The package installs a CLI script `sesasim` which you can invoke to access various functionality. 
It's recommended to run `sesasim setup` first. 
This creates a directory in your home directory which will be used for saving the GUI state. 
It's also the default location for input files and the `setup` command populates it with some default query files.

You can start the GUI mode with `sesasim gui <mode>`. To get a feel for what sesasim is all about I suggest you start
with `sesasim gui single-session`. This mode generates a single session according to the parameters you enter, 
then samples it and presents the true run times of each query and those estimated by sampling.

The `gui analyzer` mode does the same thing, except over many sessions with a range of sampling frequencies and generates
a chart of the error as a function of sampling frequency.

## Detailed instructions

The CLI has `--help` for every command, so you can access a quick description of each parameter. Some more detail is
provided below.
