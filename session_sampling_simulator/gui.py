import json
import pickle
import platform
import tkinter as tk
from os import environ
from pathlib import Path
from sys import base_prefix
from tkinter import ttk

from session_sampling_simulator.session_simulator import (
    Query,
    compare_true_and_resampled,
    generate_session,
)
import logging
from session_sampling_simulator.analysis.total_duration import (
    analyze_many_and_chart,
    get_sample_period_range,
)


logger = logging.getLogger(__name__)

if not ("TCL_LIBRARY" in environ and "TK_LIBRARY" in environ):
    try:
        tk.Tk()
    except tk.TclError:
        tk_dir = "tcl" if platform.system() == "Windows" else "lib"
        tk_path = Path(base_prefix) / tk_dir
        environ["TCL_LIBRARY"] = str(next(tk_path.glob("tcl8.*")))
        environ["TK_LIBRARY"] = str(next(tk_path.glob("tk8.*")))


def add_header(root: tk.Tk, text: str, row: int, col: int) -> None:
    ttk.Label(root, text=text, font=("Helvetica", 14, "bold")).grid(
        row=row, column=col, padx=5, pady=5, sticky="e"
    )


def add_labeled_field(
    root: tk.Tk, label: str, var, row: int, col: int, checkbox: bool = False
):
    ttk.Label(root, text=label).grid(row=row, column=col, padx=5, pady=5, sticky="e")
    if checkbox:
        ttk.Checkbutton(root, variable=var).grid(
            row=row, column=col + 1, padx=5, pady=5
        )
    else:
        ttk.Entry(root, textvariable=var).grid(row=row, column=col + 1, padx=5, pady=5)


def make_dict_from_ratios(ratios: str) -> dict[int, int]:
    raw_ratios = [int(x) for x in ratios.split(":")]
    return {w + 1: r for w, r in enumerate(raw_ratios)}


def make_ratios_from_dict(d: dict[int:int]) -> str:
    # note this won't preserve the keys, just their order
    return ":".join([str(v) for k, v in sorted(d.items(), key=lambda x: x[0])])


def extract_queries_from_vars(tk_vars: dict) -> list[Query]:
    return [
        Query(
            id=int(k[6:]),
            mean_duration=v["Mean Duration (ms)"].get(),
            duration_spread=v["Duration Spread (ms)"].get(),
            target_periodicity=v["Av. Periodicity (ms)"].get(),
            wait_state_ratios=make_dict_from_ratios(v["Wait-state Ratios"].get()),
            duration_distribution=v["Duration Distribution"].get(),
        )
        for k, v in tk_vars.items()
        if k[:5] == "Query" and v["Av. Periodicity (ms)"].get() > 0
    ]


def dump_gui_inputs(
    tk_vars: dict, path: Path, queries: list[Query] | None = None
) -> None:
    with open(path / "settings.json", "w") as f:
        # noinspection PyTypeChecker
        json.dump({k: v.get() for k, v in tk_vars["Global settings"].items()}, f)

    if not queries:
        queries = extract_queries_from_vars(tk_vars)

    with open(path / "queries.pickle", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(queries, f)


def load_gui_inputs(path: Path) -> tuple[dict, list[Query]]:
    with open(path / "settings.json", "r") as f:
        # noinspection PyTypeChecker
        global_settings = json.load(f)

    with open(path / "queries.pickle", "rb") as f:
        # noinspection PyTypeChecker
        queries = pickle.load(f)

    return global_settings, queries


def on_calculate_button_click(
    tk_vars: dict, result_text: tk.Text, save_dir: Path | None = None
) -> None:
    queries = extract_queries_from_vars(tk_vars)

    if save_dir:
        dump_gui_inputs(tk_vars, save_dir, queries)

    result = compare_true_and_resampled(
        session=generate_session(
            window_duration=tk_vars["Global settings"]["Duration (ms)"].get(),
            queries=queries,
            random_starts=tk_vars["Global settings"]["Randomize Session"].get(),
        ),
        sample_period=tk_vars["Global settings"]["Sample Period (ms)"].get(),
        sampling_strategy=tk_vars["Global settings"]["Sampling Strategy"].get(),
        use_calculated_sample_weights=tk_vars["Global settings"][
            "Use Calculated Sample Weights"
        ].get(),
    )

    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, str(result))
    result_text.config(state=tk.DISABLED)


def init_input_fields(
    widget: tk.Tk, field_definitions: dict, section_height: int
) -> dict:
    row = 0
    col = 0

    tk_vars = {}

    # Iterate over the input template and create a labelled input field for each item
    for group, fields in field_definitions.items():
        tk_vars[group] = {}
        add_header(widget, group, row, col)
        row += 1

        for label, default, var_class in fields:
            tk_vars[group][label] = var_class(value=default)
            add_labeled_field(
                widget,
                label,
                tk_vars[group][label],
                row,
                col,
                checkbox=(var_class == tk.BooleanVar),
            )
            row += 1
        # this doesn't actually enforce section_height rigidly, it always lets the group stay together
        if row > section_height:
            row = 0
            col += 3

    return tk_vars


def generate_query_field_defs(queries: list[Query]) -> dict:
    return {
        f"Query {q.id}": [
            ("Mean Duration (ms)", q.mean_duration, tk.IntVar),
            ("Duration Spread (ms)", q.duration_spread, tk.IntVar),
            ("Av. Periodicity (ms)", q.target_periodicity, tk.IntVar),
            (
                "Wait-state Ratios",
                make_ratios_from_dict(q.wait_state_ratios),
                tk.StringVar,
            ),
            ("Duration Distribution", q.duration_distribution, tk.StringVar),
        ]
        for q in queries
    }


def on_analyze_button_click(tk_vars: dict, save_dir: Path | None = None) -> None:
    queries = extract_queries_from_vars(tk_vars)

    if save_dir:
        dump_gui_inputs(tk_vars, save_dir, queries)

    sample_periods = get_sample_period_range(
        max_sample_period=tk_vars["Global settings"]["Max Sampling Period (ms)"].get(),
        min_sample_period=tk_vars["Global settings"]["Min Sampling Period (ms)"].get(),
        num_steps=tk_vars["Global settings"]["Number of Steps"].get(),
    )

    analyze_many_and_chart(
        queries=queries,
        durations=[tk_vars["Global settings"]["Duration (ms)"].get()],
        sampling_strategy=tk_vars["Global settings"]["Sampling Strategy"].get(),
        show_calculated=tk_vars["Global settings"]["Show calculated error"].get(),
        target_error=tk_vars["Global settings"]["Target error (%)"].get(),
        sample_periods=sample_periods,
        num_sessions=tk_vars["Global settings"]["Sessions Per Step"].get(),
        use_calculated_sample_weights=tk_vars["Global settings"][
            "Use Calculated Sample Weights"
        ].get(),
    )


def run_single_session_gui(
    initial_queries: list[Query] | None = None,
    save_dir: Path | None = None,
    restore: bool = False,
) -> None:
    if not initial_queries and not (restore and save_dir):
        raise RuntimeError(
            "You must provide either some initial queries or set restore and specify a save path"
        )

    input_section_height = 7

    root = tk.Tk()
    root.title("Sampling Simulator")

    default_global_settings = [
        ("Duration (ms)", 3600000, tk.IntVar),
        ("Sample Period (ms)", 1000, tk.IntVar),
        ("Randomize Session", True, tk.BooleanVar),
        ("Sampling Strategy", "u", tk.StringVar),
        ("Use Calculated Sample Weights", False, tk.BooleanVar),
    ]

    if restore:
        restored_gs, restored_queries = load_gui_inputs(save_dir)
        if not initial_queries:
            initial_queries = restored_queries
        global_settings = [
            (x[0], restored_gs.get(x[0], x[1]), x[2]) for x in default_global_settings
        ]
    else:
        global_settings = default_global_settings

    input_template = {
        "Global settings": global_settings,
    } | generate_query_field_defs(initial_queries)

    tk_vars = init_input_fields(
        widget=root,
        field_definitions=input_template,
        section_height=input_section_height,
    )

    calculate_button = ttk.Button(
        root,
        text="Calculate",
        command=lambda: on_calculate_button_click(tk_vars, result_text, save_dir),
    )
    calculate_button.grid(
        row=input_section_height + 5, column=0, columnspan=10, pady=10
    )

    result_text = tk.Text(
        root, height=32, width=150, state=tk.DISABLED, font=("Courier", 16)
    )
    result_text.grid(row=input_section_height + 6, column=0, columnspan=10, pady=10)

    root.mainloop()


# todo: consider factoring out GUI into a class to avoid repetition
def run_analyzer_gui(
    initial_queries: list[Query] | None, save_dir: Path | None, restore: bool = False
) -> None:
    input_section_height = 7

    root = tk.Tk()
    root.title("Sampling Analyzer")
    root.tk.call("tk", "scaling", 2.0)

    default_global_settings = [
        ("Duration (ms)", 3600000, tk.IntVar),
        ("Sampling Strategy", "u", tk.StringVar),
        ("Use Calculated Sample Weights", False, tk.BooleanVar),
        ("Max Sampling Period (ms)", 2000, tk.IntVar),
        ("Min Sampling Period (ms)", 200, tk.IntVar),
        ("Number of Steps", 10, tk.IntVar),
        ("Sessions Per Step", 10, tk.IntVar),
        ("Target error (%)", 10, tk.IntVar),
        ("Show calculated error", False, tk.BooleanVar),
    ]

    if restore:
        restored_gs, restored_queries = load_gui_inputs(save_dir)
        if not initial_queries:
            initial_queries = restored_queries
        global_settings = [
            (x[0], restored_gs.get(x[0], x[1]), x[2]) for x in default_global_settings
        ]
    else:
        global_settings = default_global_settings

    input_template = {
        "Global settings": global_settings,
    } | generate_query_field_defs(initial_queries)

    tk_vars = init_input_fields(
        widget=root,
        field_definitions=input_template,
        section_height=input_section_height,
    )

    analyze_button = ttk.Button(
        root,
        text="Analyze",
        command=lambda: on_analyze_button_click(tk_vars, save_dir),
    )
    analyze_button.grid(row=input_section_height + 5, column=0, columnspan=10, pady=10)

    root.mainloop()
