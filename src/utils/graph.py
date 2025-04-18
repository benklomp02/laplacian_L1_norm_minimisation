import networkx as nx
import numpy as np
import random
from itertools import combinations
from typing import Optional
import matplotlib.pyplot as plt


def _validate_inputs(num_nodes: int) -> None:
    """Validate the inputs for the graph generation function."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")


def _generate_node_coordinates(num_nodes: int) -> list[tuple[float, float]]:
    """Generate random 2D coordinates for the nodes."""
    return [(random.random(), random.random()) for _ in range(num_nodes)]


def _add_edges_to_graph(
    graph: nx.DiGraph,
    node_coordinates: list[tuple[float, float]],
    distance_threshold: float,
) -> None:
    """
    Add edges to the graph based on distance and edge probability.

    Args:
        graph (nx.DiGraph): The directed graph to which edges will be added.
        node_coordinates (list[tuple[float, float]]): Coordinates of the nodes.
        distance_threshold (float): Distance threshold for edge creation.
    """
    for source_node, target_node in combinations(range(len(node_coordinates)), 2):
        distance = np.linalg.norm(
            np.array(node_coordinates[source_node])
            - np.array(node_coordinates[target_node])
        )
        if distance < distance_threshold:
            if random.random() < 0.5:
                graph.add_edge(source_node, target_node)
            else:
                graph.add_edge(target_node, source_node)


def _visualize_graph(
    graph: nx.DiGraph, node_coordinates: list[tuple[float, float]]
) -> None:
    """Visualize the graph using matplotlib."""
    pos = {i: coord for i, coord in enumerate(node_coordinates)}
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=10,
    )
    plt.show()


def _compute(
    num_nodes: int,
    distance_threshold: Optional[float] = None,
    visualize_graph: bool = False,
) -> np.ndarray:
    """
    Generate a random directed graph with `num_nodes` nodes.

    The graph is created based on random points in a 2D space, where edges are added
    between nodes if their distance is below a threshold and a random probability check passes.

    Args:
        num_nodes (int): Number of nodes in the graph.
        distance_threshold (Optional[float]): Initial distance threshold for adding edges. Default is 1/num_nodes.
        visualize_graph (bool): Whether to visualize the graph using matplotlib. Default is False.

    Returns:
        np.ndarray: Adjacency matrix of the generated graph.
    """
    # Validate inputs
    _validate_inputs(num_nodes)

    # Generate random node coordinates
    node_coordinates = _generate_node_coordinates(num_nodes)

    # Initialize the graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # Set the initial threshold if not provided
    if distance_threshold is None:
        distance_threshold = 3 / num_nodes

    # Iteratively add edges until the graph is weakly connected
    max_iterations = 100
    for _ in range(max_iterations):
        _add_edges_to_graph(graph, node_coordinates, distance_threshold)

        if nx.is_weakly_connected(graph):
            break

        # Increase the threshold and clear edges if the graph is not connected
        distance_threshold *= 1.1
        graph.clear_edges()
    else:
        raise RuntimeError(
            "Failed to generate a weakly connected graph within the maximum iterations."
        )

    # Optionally visualize the graph
    if visualize_graph:
        _visualize_graph(graph, node_coordinates)

    # Return the adjacency matrix of the graph
    return nx.to_numpy_array(graph)


# Alias for external use
create_random_graph = _compute
