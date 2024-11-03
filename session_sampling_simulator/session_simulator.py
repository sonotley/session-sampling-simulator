import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
from math import prod
from collections import deque
import logging
from enum import Enum
from pandas import DataFrame, concat
from yaml import safe_load


GENERATOR = np.random.default_rng()

logger = logging.getLogger(__name__)


@dataclass
class Query:
    id: int
    mean_duration: int
    target_periodicity: int
    wait_state_ratios: dict[int, int]
    duration_spread: int = 0
    duration_distribution: str = "uniform"

    def __post_init__(self):
        self.normalized_wait_state_ratios = {
            k: v / sum(self.wait_state_ratios.values())
            for k, v in self.wait_state_ratios.items()
        }


@dataclass
class Runner:
    query: Query
    buffer_size: int

    def __post_init__(self):
        dist = self.query.duration_distribution.lower().strip()

        if dist == "uniform":
            self._durations = GENERATOR.integers(
                low=self.query.mean_duration - self.query.duration_spread,
                high=self.query.mean_duration + self.query.duration_spread,
                endpoint=True,
                size=self.buffer_size,
            ).tolist()
        elif dist == "exponential":
            self._durations = GENERATOR.exponential(
                scale=self.query.mean_duration,
                size=self.buffer_size,
            ).tolist()

    def execute(self, additional_id: int = 0) -> np.ndarray:
        """
        Returns an array of duration approximately _get_duration representing the state of the session during the
        execution of the query.

        The query will pass through the wait states specified in wait_states_ratios in sequence.

        """
        d = self._durations.pop()

        waits = [
            [1000 * additional_id + 10 * self.query.id + wid] * int(r * d)
            for wid, r in self.query.normalized_wait_state_ratios.items()
        ]
        logger.debug(
            f"Executed query / {[x[0] for x in waits if len(x)>0]} / {[len(x) for x in waits]}"
        )
        return np.asarray([x for y in waits for x in y])


@dataclass()
class Comparison:
    summary_1: DataFrame
    summary_2: DataFrame
    labels: tuple[str, str] = ("True", "Resampled")

    def __post_init__(self):
        self.runtime_1_col = f"Runtime {self.labels[0]}"
        self.runtime_2_col = f"Runtime {self.labels[1]}"

        self.query_wait_df = self.summary_1.merge(
            self.summary_2,
            on=["Query ID", "Wait ID"],
            suffixes=tuple(f" {x}" for x in self.labels),
            how="left",
        ).fillna(0)

        self.query_df = (
            self.query_wait_df.groupby("Query ID")
            .sum()
            .drop("Wait ID", axis=1)
            .reset_index()
        )

        self.query_wait_df["Relative Error (%)"] = (
            self.query_wait_df[self.runtime_2_col]
            / self.query_wait_df[self.runtime_1_col]
            - 1
        ) * 100

        self.query_df["Relative Error (%)"] = (
            self.query_df[self.runtime_2_col] / self.query_df[self.runtime_1_col] - 1
        ) * 100

    def __str__(self):
        query_errors = self.error_stats(by_query=True)
        query_wait_errors = self.error_stats(by_query=False)

        return (
            dataframe_to_text(self.query_wait_df)
            + "\n\n"
            + dataframe_to_text(self.query_df)
            + "\n\n"
            + f"Error by query-wait: {query_wait_errors[0]:.2f}% / {query_wait_errors[1]:.2f}% (RMS of relative error/ Relative RMS error)"
            + "\n\n"
            + f"Error by query: {query_errors[0]:.2f}% / {query_errors[1]:.2f}% (RMS of relative error/ Relative RMS error)"
        )

    def as_flat_dataframe(self):
        return concat([self.query_wait_df, self.query_df], axis=0).reset_index(
            drop=True
        )

    def error_stats(self, by_query: bool):
        comparison = self.query_df if by_query else self.query_wait_df

        r_rmse = (
            100
            * (
                (
                    (comparison[self.runtime_2_col] - comparison[self.runtime_1_col])
                    ** 2
                ).mean()
                ** 0.5
            )
            / comparison[self.runtime_1_col].mean()
        )

        rms_re = (comparison["Relative Error (%)"] ** 2).mean() ** 0.5

        return rms_re, r_rmse


class SamplingStrategy(str, Enum):
    UNIFORM = "u"
    JITTERED = "j"
    EXPONENTIAL = "e"
    POSTHOCRANDOM = "phr"


class IdStrategy(str, Enum):
    NONE = "n"
    SEQUENTIAL = "s"
    TIMESTAMP = "t"


def load_queries_from_file(path: str) -> list[Query]:
    with open(path) as f:
        contents = safe_load(f)
    return [Query(**x) for x in contents["queries"]]


def get_fill_factor(queries: list[Query]) -> float:
    """Returns a fill factor; the inverse of the estimated fraction of the session the queries will take to execute

    The fill factor is an optimisation designed to short circuit the generation of query start times once there are
    *probably* enough queries to fill the session. This is a fairly minor optimisation and the estimate is crude so
    if this causes any problematic behaviour it is generally safe to simply set fill_factor to around 1 rather than
    call this function.

    """

    query_active_fractions_est = [
        (1 / q.target_periodicity) * q.mean_duration for q in queries
    ]

    inefficiency_coefficient = 0.5
    naive_session_length = sum(
        query_active_fractions_est
    ) + inefficiency_coefficient * prod(1 - x for x in query_active_fractions_est)
    return 1 / naive_session_length


def generate_session(
    window_duration: int,
    queries: list[Query],
    random_starts: bool = True,
    id_strategy: IdStrategy = IdStrategy.TIMESTAMP,
) -> np.ndarray:
    """Generates a session array where each element represents a ms of session runtime

    The value of the element is equal to the ID of the query being executed at that moment.
    The session is populated by randomly generating 'desired' start times for each of the specified queries
    based on their target periodicity. These are combined into a single queue and the queries are executed
    sequentially at their desired start time or as soon as possible afterward if the session is already busy.

    This results in a slightly artificial session profile where it is always busier at the end, but this is probably
    inconsequential for this simulation.

    """

    fill_factor = get_fill_factor(queries)
    logger.info(f"Using fill factor {fill_factor:.3f}")

    query_schedule = []
    for q in queries:
        if random_starts:
            start_times = (
                np.cumsum(
                    GENERATOR.exponential(
                        scale=q.target_periodicity,
                        size=int(fill_factor * window_duration / q.target_periodicity),
                    )
                )
                .astype("uint32")
                .tolist()
            )
        else:
            start_times = [
                x * q.target_periodicity
                for x in range(
                    int(fill_factor * window_duration / q.target_periodicity)
                )
            ]

        query_schedule.extend([(q.id, x) for x in start_times if x < window_duration])

    logger.info(f"Generated query schedule of {len(query_schedule)} queries")

    query_counts = dict(
        zip(*np.unique([x[0] for x in query_schedule], return_counts=True))
    )
    runner_dict = {q.id: Runner(q, query_counts.get(q.id)) for q in queries}

    query_schedule.sort(key=lambda x: -x[1])
    query_schedule = deque(query_schedule)

    session_dtype = (
        np.uint64
        if id_strategy in (IdStrategy.TIMESTAMP, IdStrategy.SEQUENTIAL)
        else np.uint16
    )
    session = np.zeros(window_duration, dtype=session_dtype)

    next_free_start = 0
    additional_id = 0
    while True:
        try:
            qid, scheduled_start = query_schedule.pop()
            actual_start = max(scheduled_start, next_free_start)
            logger.debug(
                f"Next query ID:{qid}, scheduled to start at {scheduled_start}, starting at {actual_start}"
            )
        except IndexError:
            # If we run out of items in the queue, end the loop
            break
        if id_strategy == IdStrategy.TIMESTAMP:
            additional_id = actual_start
        elif id_strategy == IdStrategy.SEQUENTIAL:
            additional_id += 1
        execution = runner_dict[qid].execute(additional_id=additional_id)
        next_free_start = actual_start + len(execution)
        try:
            session[actual_start:next_free_start] = execution
        except ValueError:
            # If the query goes off the end of the session, end the loop
            break

    logger.info(f"Session full with {len(query_schedule)} queries unused.")

    return session


def get_sample_times(
    session_duration: int,
    sample_period: int,
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
    phase: float = 0,
) -> np.ndarray:
    """Returns an array of sample times spanning a range from zero to session_duration

    In all cases the target sampling period will be sample_period.

    If no optional arguments are provided, the first sample will be at zero and subsequent samples will sample_period

    """

    n_samples = int(session_duration / sample_period)

    if strategy in (SamplingStrategy.UNIFORM, SamplingStrategy.JITTERED):
        regular_times = np.fromiter(
            (x * sample_period for x in range(n_samples)),
            dtype=np.uint32,
            count=n_samples,
        )

        if strategy == SamplingStrategy.UNIFORM:
            sample_times = regular_times + int(phase * sample_period)

        else:
            # The jitter method guarantees a degree of regularity,
            # but makes the distribution of samples horribly complex mathematically
            jitter = GENERATOR.integers(
                low=-sample_period // 2,
                high=sample_period // 2,
                size=int(session_duration / sample_period),
            )

            sample_times = regular_times + jitter

    elif strategy == SamplingStrategy.EXPONENTIAL:
        sample_times = np.cumsum(
            GENERATOR.exponential(
                scale=sample_period,
                size=n_samples * 2,  # overfill to be sure we have enough
            )
        ).astype("uint32")

        sample_times = sample_times[sample_times < session_duration]

    elif strategy == SamplingStrategy.POSTHOCRANDOM:
        sample_times = GENERATOR.integers(
            low=0, high=session_duration, size=session_duration // sample_period
        ).astype("uint32")
        sample_times = np.sort(sample_times)

    logger.info(
        f"Generated {len(sample_times)} sample times. Nominal period {sample_period}, actual period {session_duration//len(sample_times)}"
    )

    return sample_times


def get_sample_weights(sample_times: np.ndarray, session_duration: int):
    backward_diff = np.diff(sample_times, prepend=0)
    forward_diff = np.diff(sample_times, append=session_duration)
    sample_weights = 0.5 * (backward_diff + forward_diff)
    logger.info(
        f"Generated {len(sample_weights)} sample weights. Mean weight {np.mean(sample_weights)}"
    )
    return sample_weights


def summarise_session(
    session: np.ndarray,
    weights: int | np.ndarray = 1,
) -> DataFrame:
    """Returns a DataFrame of how long each query ran in the session.

    The duration is multiplied by multiplier if provided.
    """

    # Count the number of occurrences of each unique combination of query ID and wait ID
    # Multiply that by the sample period (which defaults to 1 if we're not actually sampling)

    session_qwid_only = np.mod(session, 1000)

    if not isinstance(weights, np.ndarray):
        summary_tuples = [
            ((qwid % 1000) // 10, qwid % 10, runtime * weights)
            for qwid, runtime in zip(*np.unique(session_qwid_only, return_counts=True))
            if qwid > 0
        ]
    else:
        unique_qwids = np.unique(session_qwid_only)
        summary_tuples = [
            (
                (qwid % 1000) // 10,
                qwid % 10,
                np.sum(weights[session_qwid_only == qwid]),
            )
            for qwid in unique_qwids
            if qwid > 0
        ]

    header = ["Query ID", "Wait ID", "Runtime"]

    return DataFrame(summary_tuples, columns=header, dtype=np.uint64)


def dataframe_to_text(df: DataFrame) -> str:
    """Converts a dataframe to a text table"""

    # Use this hack to stop weird bug with recasting integers as floats
    return tabulate(
        df.to_numpy(dtype="object"), df.columns, tablefmt="github", floatfmt=".2f"
    )


def find_contiguous_regions(arr: np.ndarray):
    # Find the indices where the value changes
    change_points = np.diff(arr, prepend=arr[0] - 1).nonzero()[0]

    # Generate the start and end indices for contiguous regions
    starts = change_points
    ends = np.append(change_points[1:], arr.size) - 1

    # Extract the values for each contiguous region
    values = arr[starts]

    # Combine the results
    regions = list(zip(starts, ends, values))

    return regions


def compare_true_and_resampled(
    session: np.ndarray,
    sample_period: int,
    sampling_strategy: SamplingStrategy,
    sampling_phase: float = 0,
    use_calculated_sample_weights=False,
    true_summary: DataFrame | None = None,
):
    sample_times = get_sample_times(
        session_duration=len(session),
        sample_period=sample_period,
        strategy=sampling_strategy,
        phase=sampling_phase,
    )

    if use_calculated_sample_weights:
        sample_weights = get_sample_weights(sample_times, len(session))
    else:
        sample_weights = sample_period

    return Comparison(
        summary_1=true_summary
        if true_summary is not None
        else summarise_session(session),
        summary_2=summarise_session(
            session[sample_times],
            weights=sample_weights,
        ),
    )


if __name__ == "__main__":
    # Define the queries that the session will be running, don't use 0 as an ID, that's reserved for 'no query'
    queries = load_queries_from_file("../queries.example.yml")

    c = compare_true_and_resampled(
        session=generate_session(queries=queries, window_duration=3600000),
        sample_period=1000,
        sampling_strategy=SamplingStrategy.UNIFORM,
    )

    print(c)
