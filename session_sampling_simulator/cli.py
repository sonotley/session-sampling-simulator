import logging

import typer
from session_sampling_simulator.session_simulator import (
    load_queries_from_file,
    compare_true_and_resampled,
    generate_session,
    SamplingStrategy,
)
from session_sampling_simulator.gui import run_analyzer_gui, run_single_session_gui

log_format_string = "%(asctime)s | %(levelname)s | %(process)d | %(name)s | %(message)s"

logging.basicConfig(
    level="INFO",
    format=log_format_string,
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
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


@app.command()
def gui(mode: str, num_queries: int = 5):
    if mode == "single-session":
        run_single_session_gui(num_queries)
    elif mode == "analysis":
        run_analyzer_gui(num_queries)
    else:
        return f"Invalid mode {mode}"


if __name__ == "__main__":
    app()
