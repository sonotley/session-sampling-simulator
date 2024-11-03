import logging
from pathlib import Path
from typing_extensions import Annotated

import typer
from session_sampling_simulator.session_simulator import (
    load_queries_from_file,
    compare_true_and_resampled,
    generate_session,
    SamplingStrategy,
)
from session_sampling_simulator.gui import run_analyzer_gui, run_single_session_gui
from session_sampling_simulator.example_queries import default_queries

log_format_string = "%(asctime)s | %(levelname)s | %(process)d | %(name)s | %(message)s"

logging.basicConfig(
    level="INFO",
    format=log_format_string,
)

logger = logging.getLogger(__name__)

app = typer.Typer()
gui_app = typer.Typer()
app.add_typer(gui_app, name="gui")


@app.command(help="Simulate a single session and print results to the terminal")
def simulate(
    query_file: str,
    duration: int = 3600000,
    sample_period: int = 1000,
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
    randomize_session: bool = True,
    use_calculated_sample_weights: bool = False,
):
    queries = load_queries_from_file(query_file)

    comparison = compare_true_and_resampled(
        session=generate_session(
            queries=queries, window_duration=duration, random_starts=randomize_session
        ),
        sample_period=sample_period,
        sampling_strategy=sampling_strategy,
        use_calculated_sample_weights=use_calculated_sample_weights,
    )

    print(comparison)


@gui_app.command(help="A GUI mode for simulating a single session")
def single_session(
    num_queries: Annotated[
        int, typer.Option(help="Number of query inputs to render in the GUI")
    ] = 5,
):
    run_single_session_gui(num_queries)


@gui_app.command(
    help="A GUI mode for simulating many sessions and generating an error chart"
)
def analyzer(
    num_queries: Annotated[
        int, typer.Option(help="Number of query inputs to render in the GUI")
    ] = 5,
):
    run_analyzer_gui(num_queries)


@app.command(
    help="Generates a default queries.yaml file in a new directory in the specified location"
)
def setup(
    path: Annotated[
        str,
        typer.Option(
            help="Path at which to create the sesasim directory, defaults to user's home directory"
        ),
    ] = None,
):
    if not path:
        path = Path().home()
    else:
        path = Path(path)

    target_dir = path / "sesasim"
    target_dir.mkdir()

    with open(target_dir / "queries.yaml", "w") as f:
        f.write(default_queries)


if __name__ == "__main__":
    app()
