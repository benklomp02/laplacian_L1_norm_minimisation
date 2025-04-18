import matplotlib.pyplot as plt
import os
from typing import Callable, Type, List

from src.utils.graph import create_random_graph
from src.utils.errors import ErrorCalculator


def _beautiful_title(name: str) -> str:
    """
    Convert CamelCase to snake_case.

    Args:
        name (str): The CamelCase string to convert.

    Returns:
        str: The converted snake_case string.
    """
    title = "".join([" " + i.lower() if i.isupper() else i for i in name]).lstrip(" ")
    title = title.title()
    return title


def _save_fig(fig, filename: str) -> None:
    """
    Save the figure to a file.

    Args:
        fig (plt.Figure): The figure to save.
        filename (str): The filename to save the figure as.
    """
    output_dir = "public/errors"
    filename = filename.replace(" ", "_")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_dir + "/" + filename)
    plt.close(fig)


def _run(
    num_nodes: int,
    compute_basis: Callable,
    objective_class: Type,
    error_strategy: Type,
) -> float:
    """
    Calculate the average error of the basis against the Laplacian matrix.

    Args:
        num_nodes (int): Number of nodes.
        compute_basis (Callable): Function to compute the basis.
        objective_class (Type): Objective function class.
        error_strategy (Type): Error computation strategy class.

    Returns:
        float: Average error.
    """
    adjacency_matrix = create_random_graph(num_nodes, distance_threshold=0.4)
    objective_instance = objective_class(num_nodes, adjacency_matrix)
    error_calculator = ErrorCalculator(error_strategy=error_strategy())
    return error_calculator.compute_avg_error(
        num_nodes, compute_basis, objective_instance, adjacency_matrix
    )


def _run_experiment(
    node_range: List[int],
    compute_basis_list: List[Callable],
    objective_class_list: List[Type],
    error_strategy: Type,
) -> None:
    """
    Run the experiment for a given range of nodes and multiple basis computation methods.

    Args:
        node_range (List[int]): List of node counts to run the experiment on.
        compute_basis_list (List[Callable]): List of functions to compute the basis.
        objective_class_list (List[Type]): List of objective function classes.
        error_strategy (Type): Error computation strategy class.

    Returns:
        None
    """
    assert len(compute_basis_list) == len(
        objective_class_list
    ), "The number of compute_basis functions must match the number of objective classes."
    plt.figure()
    for compute_basis, objective_class in zip(compute_basis_list, objective_class_list):
        avg_errors = []
        for num_nodes in node_range:
            avg_error = _run(num_nodes, compute_basis, objective_class, error_strategy)
            avg_errors.append(avg_error)
        basis_title = _beautiful_title(objective_class.__name__) or "Unknown"
        plt.plot(node_range, avg_errors, marker="o", label=basis_title)

    plt.xlabel("N")
    plt.ylabel("Average Error")
    error_title = _beautiful_title(error_strategy.__name__)
    plt.title(f"{error_title}")
    plt.grid(True)
    plt.legend()
    _save_fig(plt.gcf(), f"avg_error_{error_title}.png".lower())
    plt.show()
    plt.close()


run_experiment = _run_experiment
