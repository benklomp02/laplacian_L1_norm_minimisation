import argparse
import logging
import time
from typing import Callable, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.graph import create_random_graph
from src.utils.pmatrix import create_random_partition_matrix
import src.tests.lib.funcs as objectives

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type alias for objective functions
objective_t = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

# Directory to save figures
SAVE_FIG_PATH = "public/figures/"


def evaluate_functions(
    funcs: Dict[str, objective_t],
    num_partitions: int,
    weights: np.ndarray,
    show_progress: bool,
) -> Dict[str, List[float]]:
    """
    Evaluate each function in `funcs` across all partitions, collect raw outputs.
    Returns a dict mapping function names to list of output values.
    """
    outputs = {name: [] for name in funcs}
    iterator = (
        tqdm(range(num_partitions), desc="Evaluating partitions", unit="part")
        if show_progress
        else range(num_partitions)
    )
    n = weights.shape[0]
    for _ in iterator:
        m = np.random.randint(2, n + 1)
        M = create_random_partition_matrix(n, m)
        a_vec = np.random.uniform(0, 1, m)
        a_vec = np.sort(a_vec)
        for name, func in funcs.items():
            outputs[name].append(func(weights=weights, M=M, a=a_vec))
    return outputs


def benchmark_and_evaluate(
    funcs: Dict[str, objective_t],
    n: int,
    num_trials: int,
    show_progress: bool,
    seed: Optional[int],
    show_graph: bool,
    is_weighted: bool,
    num_partitions: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    For each function, run trials to collect raw outputs and measure avg eval time per trial.
    Returns:
      times_dict: name -> list of avg eval times (sec per partition)
      outputs_dict: name -> all collected outputs across trials
    """
    logger.info("Generated %d partitions for n=%d", num_partitions, n)

    rng = np.random.default_rng(seed)
    times_dict = {name: [] for name in funcs}
    outputs_accum = {name: [] for name in funcs}

    for trial in range(1, num_trials + 1):
        weights = create_random_graph(
            n, seed=rng, visualize=show_graph, is_weighted=is_weighted
        )
        logger.info("Trial %d/%d", trial, num_trials)

        trial_outputs = evaluate_functions(
            funcs=funcs,
            num_partitions=num_partitions,
            weights=weights,
            show_progress=show_progress,
        )
        for name, vals in trial_outputs.items():
            outputs_accum[name].extend(vals)

        for name, func in funcs.items():
            elapsed = 0
            for _ in range(num_partitions):
                m = rng.integers(2, n + 1)
                M = create_random_partition_matrix(n, m)
                t0 = time.perf_counter()
                a_vec = np.random.uniform(0, 1, m)
                a_vec = np.sort(a_vec)
                elapsed += time.perf_counter() - t0
                func(weights=weights, M=M, a=a_vec)
            times_dict[name].append(elapsed)

    logger.info(
        "Avg eval times per partition (s): %s",
        {f: np.mean(t) for f, t in times_dict.items()},
    )
    return times_dict, outputs_accum


def plot_performance(
    times_dict: Dict[str, List[float]],
    n: int,
    title: str = "Evaluation Time Comparison",
    xlabel: str = "Function",
    ylabel: str = "Avg Time per Partition (s)",
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """
    Boxplot of per-function evaluation times, with individual trial points,
    mean (diamond) and median annotations, for dimension N.
    """
    names = list(times_dict.keys())
    data = [times_dict[nm] for nm in names]
    fig, ax = plt.subplots(figsize=figsize)
    # Boxplot with mean markers
    bp = ax.boxplot(
        data,
        labels=names,
        showmeans=True,
        meanprops=dict(
            marker="D", markeredgecolor="firebrick", markerfacecolor="firebrick"
        ),
        boxprops=dict(color="navy"),
        medianprops=dict(color="green"),
    )
    # Jitter raw points
    for i, nm in enumerate(names, 1):
        y = times_dict[nm]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(
            x, y, alpha=0.6, s=20, color="gray", label="Trials" if i == 1 else ""
        )
    # Annotate mean and median
    for i, nm in enumerate(names, 1):
        mean_v = np.mean(times_dict[nm])
        med_v = np.median(times_dict[nm])
        ax.text(i + 0.15, mean_v, f"μ={mean_v:.1e}", fontsize=9, color="firebrick")
        ax.text(i + 0.15, med_v, f"m={med_v:.1e}", fontsize=9, color="green")
    # Description text
    desc = (
        "Box: IQR; whiskers: range of trials;"
        "Gray dots: individual trials; red diamond: mean; green line: median."
    )
    fig.text(0.02, 0.02, desc, fontsize=8, va="bottom")
    ax.set_title(f"{title} (N={n})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend(loc="upper right")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fn = f"{SAVE_FIG_PATH}performance_plot.png"
    fig.savefig(fn)
    plt.close(fig)
    logger.info("Saved performance plot as '%s'", fn)


def plot_output_distributions(
    outputs_dict: Dict[str, List[float]],
    n: int,
    bins: int = 60,
    figsize: Tuple[int, int] = (10, 5),
    density: bool = True,
) -> None:
    plt.figure(figsize=figsize)
    for name, outputs in outputs_dict.items():
        plt.hist(
            outputs,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=name,
        )
    plt.title(f"Output Distributions (Normalized, N={n})")
    plt.xlabel("Objective Value")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fn = f"{SAVE_FIG_PATH}output_distributions.png"
    plt.savefig(fn)
    plt.close()
    logger.info("Saved output distributions plot as '%s'", fn)


def plot_error_distributions(
    outputs_dict: Dict[str, List[float]],
    baseline: str,
    n: int,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Histogram of *relative* error vs. baseline, in percent:
        100 * (f_approx - f_original) / f_original
    """
    plt.figure(figsize=figsize)
    base = np.array(outputs_dict[baseline])
    for name, outputs in outputs_dict.items():
        if name == baseline:
            continue
        arr = np.array(outputs)
        rel_err = 100.0 * (arr - base) / base
        plt.hist(
            rel_err,
            bins=bins,
            alpha=0.6,
            label=f"{name} vs {baseline}",
        )
    plt.axvline(0, color="k", linestyle="--")
    plt.title(f"Relative Error Distributions vs. {baseline} (N={n})")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    fn = f"{SAVE_FIG_PATH}relative_error_distributions.png"
    plt.savefig(fn)
    plt.close()
    logger.info("Saved relative error distributions plot as '%s'", fn)


def plot_time_vs_error(
    times_dict: Dict[str, List[float]],
    outputs_dict: Dict[str, List[float]],
    baseline: str,
    n: int,
    figsize: Tuple[int, int] = (6, 5),
) -> None:
    """
    Scatter plot of mean eval time vs. mean *relative* error (%)
    for each approximation, with the baseline at (baseline_time, 0).
    """
    # Baseline outputs
    base_out = np.array(outputs_dict[baseline])
    # Mean times
    means_time = {name: np.mean(times) for name, times in times_dict.items()}
    # Mean relative error (%) for each non-baseline
    means_rel_err = {}
    for name, outputs in outputs_dict.items():
        if name == baseline:
            continue
        out = np.array(outputs)
        rel_err_pct = np.abs(out - base_out) / base_out * 100.0
        means_rel_err[name] = np.mean(rel_err_pct)

    base_time = means_time[baseline]

    plt.figure(figsize=figsize)
    # Plot baseline point at 0% error
    plt.scatter(
        base_time,
        0.0,
        color="red",
        marker="x",
        s=100,
        label=f"{baseline} (0%)",
    )
    plt.axvline(base_time, color="red", linestyle="--", linewidth=1)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    # Plot each approximation
    for name, err in means_rel_err.items():
        t = means_time[name]
        plt.scatter(t, err, s=80, label=name)
        plt.text(
            t,
            err,
            f"{err:.1f}%",
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    plt.xlabel("Mean Eval Time per Partition (s)")
    plt.ylabel("Mean Relative Error (%)")
    plt.title(f"Speed–Accuracy Tradeoff (Relative Error, N={n})")
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fn = f"{SAVE_FIG_PATH}time_vs_relative_error.png"
    plt.savefig(fn)
    plt.close()
    logger.info("Saved time vs relative error plot as '%s'", fn)


def main():
    parser = argparse.ArgumentParser(
        description="Performance & approximation evaluator with rich plots."
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show progress bar for evaluations"
    )
    parser.add_argument("--n", type=int, default=60, help="Number of nodes")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-partitions", type=int, default=100)
    parser.add_argument("--show-graph", action="store_true", default=False)
    parser.add_argument("--weighted", action="store_true", default=False)
    parser.add_argument(
        "--sample-size", type=int, default=20, help="Sample size for approx"
    )
    parser.add_argument(
        "--plot-performance",
        action="store_true",
        help="Boxplot eval times",
        default=False,
    )
    parser.add_argument(
        "--plot-outputs",
        action="store_true",
        help="Overlay output histograms",
        default=False,
    )
    parser.add_argument(
        "--plot-errors",
        action="store_true",
        help="Histograms of error vs baseline",
        default=False,
    )
    parser.add_argument(
        "--plot-tradeoff",
        action="store_true",
        help="Scatter time vs error",
        default=False,
    )
    args = parser.parse_args()

    funcs: Dict[str, objective_t] = {
        "original": objectives.f_original,
        "triangle inequality": objectives.f_approx_triangle_inequality,
        "appr. by mean": objectives.f_approx_sign_by_median,
        "appr. by weights": objectives.f_approx_sign_by_weighted,
        "appr. by median": objectives.f_approx_sign_by_mean,
    }
    baseline = "original"

    times_dict, outputs_dict = benchmark_and_evaluate(
        funcs,
        show_progress=args.progress,
        n=args.n,
        num_trials=args.trials,
        seed=args.seed,
        show_graph=args.show_graph,
        is_weighted=args.weighted,
        num_partitions=args.num_partitions,
    )

    orig_times = times_dict.get("original", [])
    if orig_times:
        logger.info(
            "Original function timing — mean: %.3e s, min: %.3e s, max: %.3e s",
            np.mean(orig_times),
            np.min(orig_times),
            np.max(orig_times),
        )

    if args.plot_performance:
        plot_performance(times_dict, args.n)
    if args.plot_outputs:
        plot_output_distributions(outputs_dict, args.n)
    if args.plot_errors:
        plot_error_distributions(outputs_dict, baseline, args.n)
    if args.plot_tradeoff:
        plot_time_vs_error(times_dict, outputs_dict, baseline, args.n)

    logger.info("Completed trials. Functions evaluated: %s", list(funcs.keys()))


if __name__ == "__main__":
    main()
