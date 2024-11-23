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

### Manual installation

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

### Single-session GUI mode

```commandline
Usage: sesasim gui single-session [OPTIONS]

 A GUI mode for simulating a single session
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --query-file                     TEXT     Path to YAML file defining the     │
│                                           queries to simulate                │
│                                           [default: None]                    │
│ --num-queries                    INTEGER  Number of query inputs to render   │
│                                           in the GUI                         │
│                                           [default: None]                    │
│ --restore        --no-restore             Restore previous inputs, ignores   │
│                                           any query file specified.          │
│                                           [default: no-restore]              │
│ --help                                    Show this message and exit.        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

If you have run `sesasim setup` you can run this command with no options and it will use a default set of queries.
If you have not run `setup` you must provide a query file and the `--restore` option will not work because your GUI inputs
will not be saved.

If you provide a query file _and_ specify `--num-queries n` only the first `n` queries of the query file will be used.
If `n` is greater than the number of queries in the file, the last query will be repeated to reach `n`.

#### GUI inputs: Global Settings

 - Duration: the duration of the session to generate in milliseconds
 - Sample Period: the average interval between samples in milliseconds
 - Randomize Session: if unchecked, query times will not be randomized but instead will start with exactly the specified periodicity (unless another query is already running)
 - Sampling Strategy: determines how the simulation decides when to take a sample.
   - u == uniform: a sample will be taken every _P_ milliseconds exactly, where _P_ is the sample period.
   - e == exponential: the probability of a given millisecond being sampled is 1/_P_ resulting in randomly spaced samples with an average period of _P_.
   - j == jittered: like uniform, but with a random jitter applied to each sample. This means the sample period is uniformly distributed between 1 and 2*P*-1.
   - phr == post-hoc random: a fraction 1/_P_ of all the milliseconds in the session are selected to be sampled. This is mathematically equivalent to exponential, but unlike the other methods, couldn't be applied to streaming data in real-time as you need to have the whole session.
 - Use Calculated Sample Weights: if checked, the weight of each sample will be calculated as half the elapsed time since the previous sample, plus half the time until the next one. If not checked each sample is equally weighted.

#### GUI inputs: Queries

For each query, you can specify the following:

 - Mean Duration: the mean duration of that query in milliseconds
 - Duration Spread: a measure of how much the duration varies, how this is used depends on _Duration Distribution_.
 - Average Periodicity: the average interval between executions of the query in milliseconds. This interval is generated from an exponential distribution function.
 - Wait-state Ratios: a ratio describing how much of the query's execution is spent in each wait state. For example `2:0:1` means that the query spends two thirds of its time in wait 1, none in wait 2 and one third it wait 3.
 - Duration Distribution: can be set to 'uniform' or 'exponential'. If set to 'uniform' the duration of each execution is drawn from a uniform distribution of width 2 x _Duration Spread_ around _Average Periodicity_. If set to 'exponential', it is drawn from an exponential distribution of mean _Average Periodicity_.
