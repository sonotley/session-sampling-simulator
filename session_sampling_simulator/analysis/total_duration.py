from time import perf_counter
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
import seaborn as sns
from pandas import concat, unique
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from humanize import precisedelta

from session_sampling_simulator.session_simulator import (
    load_queries_from_file,
    compare_true_and_resampled,
    generate_session,
    summarise_session,
    SamplingStrategy,
    Query,
    IdStrategy,
)


def get_sample_period_range(
    min_sample_period: int, max_sample_period: int, num_steps: int
) -> np.ndarray:
    """Returns an array of sample periods spread linearly in frequency space"""

    sample_divisor_limit = max_sample_period / min_sample_period
    sample_divisors = np.linspace(
        start=1, stop=sample_divisor_limit, num=num_steps, endpoint=True
    )
    return max_sample_period / sample_divisors


def analyse_one_session(
    queries: list,
    duration: int,
    sampling_strategy: SamplingStrategy,
    sample_periods: np.ndarray,
    use_calculated_sample_weights: bool,
):
    # This is a great hack for faster error analysis with uniform sampling
    # but for other strategies it just re-runs the same session
    # However given that all other strategies use random sample times, this is probably ok
    phase_divisions = 20

    runs = []
    session = generate_session(duration, queries, id_strategy=IdStrategy.NONE)
    true_summary = summarise_session(session)
    for sp in sample_periods:
        for phase in (x / phase_divisions for x in range(phase_divisions)):
            comparison = compare_true_and_resampled(
                sample_period=round(sp),
                sampling_strategy=sampling_strategy,
                session=session,
                true_summary=true_summary,
                sampling_phase=phase,
                use_calculated_sample_weights=use_calculated_sample_weights,
            )

            run_df = comparison.as_flat_dataframe()
            run_df["sample_period"] = sp
            run_df["Sampling Frequency / Hz"] = 1000 / sp
            run_df["Sampling"] = sampling_strategy
            run_df["Session Duration"] = duration
            runs.append(run_df)

    return concat(runs)


def analyze_many_and_chart(
    queries: list,
    durations: list[int],
    sampling_strategy: SamplingStrategy,
    use_calculated_sample_weights: bool,
    show_calculated: bool,
    target_error: int,
    sample_periods: np.ndarray,
    num_sessions: int = 10,
    chart_themes: tuple = ("light",),
    save_dir: Path | None = None,
    chart_type: str = "frequency-query",
):
    plt.close()
    tic = perf_counter()

    pool = Pool(processes=min(os.cpu_count(), num_sessions * len(durations)))
    session_results = pool.starmap(
        analyse_one_session,
        [
            (
                queries,
                d,
                sampling_strategy,
                sample_periods,
                use_calculated_sample_weights,
            )
            for d in durations
        ]
        * num_sessions,
    )

    results = concat(session_results)

    print(perf_counter() - tic)

    for theme in chart_themes:
        show_chart(
            results=results,
            target_error=target_error,
            chart_theme=theme,
            chart_type=chart_type,
        )
        if show_calculated:
            add_calculated_error_line(
                query=queries[0],
                min_sampling_period=sample_periods.min(),
                max_sampling_period=sample_periods.max(),
                duration=durations[0],
            )

        if save_dir:
            plt.savefig(save_dir / f"analysis-{theme}.svg")

        plt.show()


def show_chart(
    results: DataFrame,
    target_error: int,
    chart_type: str = "frequency-query",
    chart_theme: str = "light",
    force_zero: bool = True,
):
    if chart_theme == "dark":
        background = "#232425"
        inner_lines = "#a9a9b3"
        outer_lines = inner_lines
    elif chart_theme == "light":
        inner_lines = "#232425"
        background = "white"
        outer_lines = "black"

    chart_palette = ["#cfe2f3ff", "#d9ead3ff", "#d9d2e9ff", "#fff2ccff", "#f4ccccff"]

    sns.set_style(
        rc={
            "figure.facecolor": background,
            "axes.facecolor": background,
            "grid.color": inner_lines,
            "lines.color": inner_lines,
            "text.color": inner_lines,
            "xtick.color": outer_lines,
            "ytick.color": outer_lines,
            "axes.labelcolor": outer_lines,
        }
    )

    ax = plt.gca()  # Get current axes
    ax.spines["top"].set_color(outer_lines)
    ax.spines["bottom"].set_color(outer_lines)
    ax.spines["left"].set_color(outer_lines)
    ax.spines["right"].set_color(outer_lines)

    if chart_type == "frequency-query":
        results = results[results["Wait ID"].isna()]
        x_axis_var = "Sampling Frequency / Hz"
        grouping_var = "Query ID"
        discrete_x = False
    elif chart_type == "frequency-wait":
        results = results[~results["Wait ID"].isna()]
        x_axis_var = "Sampling Frequency / Hz"
        grouping_var = "Query ID"
        discrete_x = False
    elif chart_type == "duration-frequency":
        results = results[results["Wait ID"].isna()]
        x_axis_var = "Session Duration"
        grouping_var = "Sampling Frequency / Hz"
        discrete_x = True
        ax.set_xticklabels(
            [precisedelta(x / 1000) for x in unique(results[x_axis_var])]
        )

    else:
        raise ValueError(f"Unknown chart type {chart_type}")

    hue_params = {"hue": grouping_var, "palette": chart_palette}

    chart = sns.boxplot(
        results,
        x=x_axis_var,
        y="Relative Error (%)",
        native_scale=not discrete_x,
        whis=(5, 95),  # type: ignore
        fliersize=0,
        color="#cfe2f3ff",
        linecolor=inner_lines,
        **hue_params,
    )

    if target_error > 0:
        chart.axhline(target_error, linestyle="--")
        chart.axhline(-target_error, linestyle="--")

    m = 1.1
    # Assume that the left-most x-axis variable contains the group with the largest error
    results_left_and_large = results[
        results[x_axis_var] == results[x_axis_var].min()
    ].groupby(grouping_var)["Relative Error (%)"]
    lower_lim = m * results_left_and_large.quantile(0.05).min()
    upper_lim = m * results_left_and_large.quantile(0.95).max()

    if force_zero:
        lim = max(abs(lower_lim), abs(upper_lim))
        plt.ylim(-lim, lim)
    else:
        plt.ylim((lower_lim, upper_lim))

    fig = plt.gcf()
    fig.set_size_inches(8, 4)


def add_calculated_error_line(
    query: Query,
    duration: int,
    max_sampling_period: int,
    min_sampling_period: int,
):
    x_fun = np.linspace(1000 / max_sampling_period, 1000 / min_sampling_period, 200)
    # TODO: add another line that uses observed P and not expected P here? How would I deal with P being per session?
    p_exp = query.mean_duration * (1 / query.target_periodicity)

    y_fun = 168 * np.sqrt(1000 * (1 - p_exp) / (x_fun * duration * p_exp))

    # Finite population correction where the population is the milliseconds in the sample
    fp_corr1 = np.sqrt(1 - (x_fun / 1000))
    # weird fpc based on proportion of queries sampled, not quite right but better than the above
    # fp_corr2 = np.sqrt(1 - (query.mean_duration * x_fun / 1000))
    y_fun1 = fp_corr1 * y_fun
    # y_fun2 = fp_corr2 * y_fun

    plt.plot(x_fun, y_fun1, linestyle=":", color="#fe5186")
    plt.plot(x_fun, -y_fun1, linestyle=":", color="#fe5186")
    # plt.plot(x_fun, y_fun2, linestyle=":", color="orange")
    # plt.plot(x_fun, -y_fun2, linestyle=":", color="orange")


if __name__ == "__main__":
    queries = load_queries_from_file("../../queries.example.yml")
    analyze_many_and_chart(
        queries,
        durations=[3600000],
        sampling_strategy=SamplingStrategy.UNIFORM,
        use_calculated_sample_weights=False,
        show_calculated=True,
        target_error=10,
        sample_periods=get_sample_period_range(
            max_sample_period=5000, min_sample_period=100, num_steps=15
        ),
        num_sessions=5,
        chart_themes=("light", "dark"),
        save_dir=Path("../../local"),
        chart_type="frequency-query",
    )

    # analyze_many_and_chart(
    #     queries,
    #     durations=[3600000, 3600000*6, 3600000*12, 3600000*24*3],
    #     sampling_strategy=SamplingStrategy.UNIFORM,
    #     use_calculated_sample_weights=False,
    #     show_calculated=False,
    #     target_error=10,
    #     sample_periods=np.asarray([50000, 5000, 1000, 100]),
    #     num_sessions=1,
    #     chart_themes=("light", "dark"),
    #     save_dir=Path("../../local"),
    #     chart_type="duration-frequency",
    # )
