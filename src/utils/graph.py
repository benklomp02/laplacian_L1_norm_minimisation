import networkx as nx
import numpy as np
from itertools import combinations
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt


def _validate_inputs(num_nodes: int) -> None:
    """Validate the inputs for the graph generation function."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")


def _add_edges_to_graph(
    graph: nx.DiGraph,
    coords: np.ndarray,
    distance_threshold: float,
    rng: np.random.Generator,
) -> None:
    """
    Add directed edges based on spatial distance and random orientation.
    """
    n = coords.shape[0]
    for i, j in combinations(range(n), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < distance_threshold:
            # random orientation
            if rng.random() < 0.5:
                graph.add_edge(i, j)
            else:
                graph.add_edge(j, i)


def _visualize_graph(
    graph: nx.DiGraph,
    coords: np.ndarray,
) -> None:
    """Visualize the directed graph using matplotlib."""
    pos = {i: tuple(coords[i]) for i in range(coords.shape[0])}
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=10,
        arrowsize=15,
    )
    if nx.is_weighted(graph):
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.show()


def _add_random_weights_to_graph(
    graph: nx.DiGraph,
) -> None:
    """
    Add weights to the edges of the graph.
    """
    for u, v in graph.edges():
        graph[u][v]["weight"] = np.random.randint(
            1, 10
        )  # Random weight between 1 and 10


def create_random_graph(
    num_nodes: int,
    distance_threshold: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    visualize: bool = False,
    is_weighted: bool = True,
) -> np.ndarray:
    """
    Generate a weakly connected random directed graph.

    Args:
        num_nodes: number of nodes in the graph.
        distance_threshold: initial distance cutoff (default: 3/num_nodes).
        seed: random seed (int) or np.random.Generator for reproducibility.
        visualize: if True, display a plot of the generated graph.

    Returns:
        Adjacency matrix (np.ndarray) of shape (num_nodes, num_nodes).
    """
    _validate_inputs(num_nodes)

    # Initialize RNG
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Generate random 2D coordinates
    coords = rng.random((num_nodes, 2))

    # Initialize directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # Set default threshold
    if distance_threshold is None:
        distance_threshold = 3 / num_nodes

    # Try until weakly connected or timeout
    max_iter = 100
    thr = distance_threshold
    for _ in range(max_iter):
        graph.clear_edges()
        _add_edges_to_graph(graph, coords, thr, rng)
        if nx.is_weakly_connected(graph):
            break
        thr *= 1.1
    else:
        raise RuntimeError(
            f"Failed to generate weakly connected graph after {max_iter} iterations"
        )
    if is_weighted:
        _add_random_weights_to_graph(graph)
    # Optional visualization
    if visualize:
        _visualize_graph(graph, coords)

    # Return adjacency matrix
    return nx.to_numpy_array(graph)
