import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Callable, List, Type
import shutil
import os
import math

from src.utils.graph import create_random_graph
from src.utils.objectives import Objective


def _create_fresh_directory():
    """
    Create a fresh directory for saving figures.
    """

    dir_path = "public/basis_vectors"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def visualize_basis_vectors(
    compute_basis: Callable[[int, Objective], np.ndarray],
    objective_classes: List[Type[Objective]],
    num_nodes: int = 30,
    basis_indices: List[int] = [0, 1, 2],
    layout: str = "spring",
    annotate_values: bool = False,
    save_path: str = None,
    max_objectives: int = 4,
) -> None:
    """
    Create a random graph, compute bases for multiple objective classes,
    and visualize selected basis vectors in a 2-column grid layout.

    Args:
        compute_basis: Function that computes the orthonormal basis.
        objective_classes: List of Objective classes.
        num_nodes: Number of nodes in the graph.
        basis_indices: Indices of basis vectors to visualize.
        layout: Layout algorithm to use ("spring", "kamada", "circular", "spectral").
        annotate_values: Whether to show numeric values on nodes.
        save_path: Directory path to save figures. If None, plots are shown.
        max_objectives: Maximum number of objective functions to display.
    """
    objective_classes = objective_classes[:max_objectives]

    # Step 1: Create graph
    adjacency_matrix = create_random_graph(num_nodes, distance_threshold=0.4)
    G = nx.from_numpy_array(adjacency_matrix)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    if save_path:
        _create_fresh_directory()

    for i in basis_indices:
        n_cols = 2
        n_rows = int(np.ceil(len(objective_classes) / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axs = np.array(axs).reshape(-1)  # Flatten in case of 2D

        for j, objective_class in enumerate(objective_classes):
            ax = axs[j]
            objective = objective_class(num_nodes, adjacency_matrix)
            basis = compute_basis(num_nodes, objective)
            b_i = basis[:, i]
            vmin, vmax = -np.max(np.abs(b_i)), np.max(np.abs(b_i))

            # Compute graph smoothness: x^T L x
            L = nx.laplacian_matrix(G).toarray()
            smoothness = float(b_i.T @ L @ b_i)

            # Compute sparsity: fraction of near-zero entries
            zero_thresh = 1e-3
            sparsity = np.sum(np.abs(b_i) < zero_thresh) / len(b_i)

            # Compute L1 variation
            l1_variation = np.sum(np.abs(np.diff(b_i)))

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
            nodes = nx.draw_networkx_nodes(
                G,
                pos,
                node_color=b_i,
                cmap=plt.cm.coolwarm,
                node_size=200,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

            if annotate_values:
                for node, value in enumerate(b_i):
                    ax.text(
                        pos[node][0],
                        pos[node][1] + 0.03,
                        f"{value:.2f}",
                        fontsize=7,
                        ha="center",
                        color="black",
                    )

            ax.set_title(f"{objective_class.__name__}", fontsize=10)

            # Draw info box
            info_str = f"Smoothness: {smoothness:.2f}\nSparsity: {sparsity:.2f}\nL1 Variation: {l1_variation:.2f}"
            bbox_props = dict(
                boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7
            )
            ax.text(
                0.05,
                0.95,
                info_str,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=bbox_props,
            )
            bbox_props = dict(
                boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7
            )
            ax.text(
                0.05,
                0.95,
                info_str,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=bbox_props,
            )
            ax.axis("off")
            fig.colorbar(nodes, ax=ax)
            ax.axis("off")

        # Hide unused subplots
        for j in range(len(objective_classes), len(axs)):
            axs[j].axis("off")

        fig.suptitle(f"Basis Vector {i}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            filename = f"grid_basis_vector_{i}.png"
            fig.savefig(os.path.join(save_path, filename))
            plt.close(fig)
        else:
            plt.show()


def main():
    from src.algorithms.greedy import compute_basis
    from src.utils.objectives import Objective0, Objective1, Objective2, Objective3

    SAVE_PATH = "public/basis_vectors"
    n = 40
    visualize_basis_vectors(
        compute_basis,
        [Objective0, Objective1, Objective2, Objective3],
        num_nodes=n,
        basis_indices=[0, 1, 2] + list(range(3, n, 3)),
        layout="kamada",
        annotate_values=False,
        save_path=SAVE_PATH,
    )


if __name__ == "__main__":
    main()
