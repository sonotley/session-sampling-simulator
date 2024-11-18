import logging
from copy import copy
from pathlib import Path
from typing_extensions import Annotated

import typer
from session_sampling_simulator.session_simulator import (
    load_queries_from_file,
    compare_true_and_resampled,
    generate_session,
    SamplingStrategy,
    Query,
)
from session_sampling_simulator.gui import run_analyzer_gui, run_single_session_gui
from session_sampling_simulator.default_queries import (
    queries_yaml,
)
from session_sampling_simulator.analysis.total_duration import (
    analyze_many_and_chart,
    get_sample_period_range,
)

log_format_string = "%(asctime)s | %(levelname)s | %(process)d | %(name)s | %(message)s"

logging.basicConfig(
    level="INFO",
    format=log_format_string,
)

logger = logging.getLogger(__name__)

DEFAULT_FILE_LOCATION = Path().home()
DEFAULT_DIR_NAME = "sesasim"
DEFAULT_QUERY_FILE = "queries.yaml"
DEFAULT_GUI_QUERY_FILE = ".gui_defaults.yaml"
DEFAULT_DIR_PATH = DEFAULT_FILE_LOCATION / DEFAULT_DIR_NAME
DEFAULT_FULL_PATH = DEFAULT_DIR_PATH / DEFAULT_QUERY_FILE
DEFAULT_GUI_FULL_PATH = DEFAULT_DIR_PATH / DEFAULT_GUI_QUERY_FILE

app = typer.Typer()
gui_app = typer.Typer()
app.add_typer(gui_app, name="gui")


def duplicate_query(
    base_query: Query, n: int, include_base_id: bool = False
) -> list[Query]:
    query_list = [copy(base_query) for _ in range(n)]
    for i, q in enumerate(query_list):
        q.id = i + base_query.id + 1 if not include_base_id else 0
    return query_list


def pad_or_truncate_query_list(queries: list[Query], n: int) -> list[Query]:
    if len(queries) == n:
        return queries
    if len(queries) > n:
        return queries[:n]

    return queries + duplicate_query(queries[-1], n - len(queries))


@app.command(help="Simulate a single session and print results to the terminal")
def simulate(
    query_file: Annotated[
        str, typer.Option(help="Path to YAML file defining the queries to simulate")
    ] = DEFAULT_FULL_PATH,
    duration: Annotated[
        int, typer.Option(help="Duration of session to simulate in milliseconds")
    ] = 3600000,
    sample_period: Annotated[
        int, typer.Option(help="Average interval between samples in milliseconds")
    ] = 1000,
    sampling_strategy: Annotated[
        SamplingStrategy, typer.Option(help="Strategy used to determine when to sample")
    ] = SamplingStrategy.UNIFORM,
    randomize_session: Annotated[
        bool, typer.Option(help="Start queries at random times")
    ] = True,
    use_calculated_sample_weights: Annotated[
        bool,
        typer.Option(
            help="Use the actual sample interval for each sample rather than the nominal period when estimating total runtime."
        ),
    ] = False,
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


@app.command(
    help="Simulate many sessions and show or save a chart of the sampling error"
)
def analyze(
    query_file: Annotated[
        str, typer.Option(help="Path to YAML file defining the queries to simulate")
    ] = DEFAULT_FULL_PATH,
    duration: Annotated[
        int, typer.Option(help="Duration of session to simulate in milliseconds")
    ] = 3600000,
    max_sample_period: Annotated[
        int, typer.Option(help="Maximum sample period to use")
    ] = 1000,
    min_sample_period: Annotated[
        int, typer.Option(help="Maximum sample period to use")
    ] = 100,
    steps: Annotated[
        int,
        typer.Option(
            help="Number of sample period increments to test between max and min (inclusive)",
            min=2,
        ),
    ] = 10,
    sampling_strategy: Annotated[
        SamplingStrategy, typer.Option(help="Strategy used to determine when to sample")
    ] = SamplingStrategy.UNIFORM,
    use_calculated_sample_weights: Annotated[
        bool,
        typer.Option(
            help="Use the actual sample interval for each sample rather than the nominal period when estimating total runtime."
        ),
    ] = False,
    target_error: Annotated[
        int, typer.Option(help="Target error in percent to show on the chart")
    ] = 0,
    show_expected_error: Annotated[
        bool,
        typer.Option(
            help="Show expected error calculated from the session parameters for query_id=1"
        ),
    ] = False,
):
    queries = load_queries_from_file(query_file)

    analyze_many_and_chart(
        queries=queries,
        durations=[duration],
        sampling_strategy=sampling_strategy,
        use_calculated_sample_weights=use_calculated_sample_weights,
        show_calculated=show_expected_error,
        target_error=target_error,
        sample_periods=get_sample_period_range(
            min_sample_period, max_sample_period, steps
        ),
    )


@gui_app.command(help="A GUI mode for simulating a single session")
def single_session(
    query_file: Annotated[
        str, typer.Option(help="Path to YAML file defining the queries to simulate")
    ] = None,
    num_queries: Annotated[
        int | None, typer.Option(help="Number of query inputs to render in the GUI")
    ] = None,
    restore: Annotated[
        bool,
        typer.Option(help="Restore previous inputs, ignores any query file specified."),
    ] = False,
):
    save_dir = (
        DEFAULT_DIR_PATH / "gui-single-session" if DEFAULT_DIR_PATH.exists() else None
    )

    if restore:
        run_single_session_gui(save_dir=save_dir, restore=restore)
    else:
        if not query_file:
            query_file = DEFAULT_GUI_FULL_PATH

        queries = load_queries_from_file(query_file)

        if num_queries:
            queries = pad_or_truncate_query_list(queries, num_queries)

        run_single_session_gui(queries, save_dir=save_dir, restore=restore)


@gui_app.command(
    help="A GUI mode for simulating many sessions and generating an error chart"
)
def analyzer(
    # mode: Annotated[
    #     str,
    #     typer.Option(
    #         help="Mode, one of 'frequency-query', 'frequency-waits' or 'duration-frequency'"
    #     ),
    # ] = "frequency-query",
    query_file: Annotated[
        str, typer.Option(help="Path to YAML file defining the queries to simulate")
    ] = DEFAULT_GUI_FULL_PATH,
    num_queries: Annotated[
        int | None, typer.Option(help="Number of query inputs to render in the GUI")
    ] = None,
    restore: Annotated[
        bool,
        typer.Option(help="Restore previous inputs, ignores any query file specified."),
    ] = False,
):
    save_dir = DEFAULT_DIR_PATH / "gui-analyzer" if DEFAULT_DIR_PATH.exists() else None

    if restore:
        run_analyzer_gui(save_dir=save_dir, restore=restore)
    else:
        if not query_file:
            query_file = DEFAULT_GUI_FULL_PATH

        queries = load_queries_from_file(query_file)

        if num_queries:
            queries = pad_or_truncate_query_list(queries, num_queries)

        run_analyzer_gui(queries, save_dir=save_dir, restore=restore)


# todo: add an analyzer mode switch or another command


@app.command(
    help="Generates default query files in a new directory in user's home directory"
)
def setup():
    target_dir = DEFAULT_FILE_LOCATION / DEFAULT_DIR_NAME
    target_dir.mkdir()
    (target_dir / "gui-single-session").mkdir()
    (target_dir / "gui-analyzer").mkdir()

    with open(target_dir / DEFAULT_QUERY_FILE, "w") as f:
        f.write(queries_yaml)

    with open(target_dir / DEFAULT_GUI_QUERY_FILE, "w") as f:
        f.write(queries_yaml)


if __name__ == "__main__":
    app()
