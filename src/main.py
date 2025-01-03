from rustworkx import PyDiGraph
from typing import Optional


def total_degree(graph: PyDiGraph, index: int) -> int:
    return graph.in_degree(index) + graph.out_degree(index)


def neighbors_degrees(graph: PyDiGraph, index: int, reverse: bool = False) -> list[int]:
    return sorted(
        [total_degree(graph, n) for n in graph.neighbors(index)], reverse=reverse
    )


def degree_and_neighbors_degrees(graph: PyDiGraph, index: int) -> tuple[int, list[int]]:
    """Sort by degree of node and then by the neighbor degree sequence"""
    neighbors = neighbors_degrees(graph, index)
    return (graph.in_degree(index) + graph.out_degree(index), neighbors)


def can_support(
    device_graph: PyDiGraph,
    device_node_index: int,
    query: PyDiGraph,
    query_node_index: int,
) -> bool:
    """It needs to check that
    - The in_degree and the out_dgree must the be same.
    - The degree of device node's neighbors must be the same as the degree of the query node's neighbors(not sure about this one)
    """
    query_neighbors = neighbors_degrees(query, query_node_index)

    device_node_neighbors = neighbors_degrees(device_graph, device_node_index)

    return (
        device_graph.in_degree(device_node_index) == query.in_degree(query_node_index)
        and device_graph.out_degree(device_node_index)
        == query.out_degree(query_node_index)
        and query_neighbors == device_node_neighbors
    )


def check_mapping(
    most_constrained: int,
    device_node: int,
    query: PyDiGraph,
    device_graph: PyDiGraph,
    current_mappings: dict[int, int],
) -> bool:
    """
    m: the most contrained query node.
    n: a neighbor in the values of the current mappings
    D: the keys in the current_mappings
    f(x): the values in the current_mappings

    If there is a neighbor d ∈ D of m such that n is not neighbors with f(d),
    or if there is a non-neighbor d ∈ D of m such that n is neighbors with f (d)
    [or if assigning f (m) = n would violate a symmetry-breaking condition in C(h)], then continue with the next n.
    """
    neighbors_most_contrained = set(query.neighbors(most_constrained))
    device_node_neighbors = set(device_graph.neighbors(device_node))

    # If there is a neighbor d ∈ D of m such that n is not neighbors with f(d),
    neighbors_in_mapping = neighbors_most_contrained.intersection(
        current_mappings.keys()
    )
    device_nodes = {current_mappings[d] for d in neighbors_in_mapping}
    if device_nodes.isdisjoint(device_node_neighbors):
        return False
    # or if there is a non-neighbor d ∈ D of m such that n is neighbors with f (d)
    non_neighbors_most_constrained = set(current_mappings.keys()).intersection(
        neighbors_most_contrained
    )
    device_nodes = {current_mappings[d] for d in non_neighbors_most_constrained}
    if not device_nodes.isdisjoint(device_node_neighbors):
        return False

    return True


def isomporphic_extension(
    map_query_to_device: dict[int, int], query: PyDiGraph, device_graph: PyDiGraph
) -> Optional[dict[int, int]]:
    if map_query_to_device.keys() == set(query.node_indices()):
        return map_query_to_device

    all_non_mapped_neighbors = sorted(
        [
            (
                neighbor,
                total_degree(query, neighbor),
            )  ## TODO: need to constrain on neihbors mapped
            for node in map_query_to_device.keys()
            for neighbor in query.neighbors(node)
            if neighbor not in map_query_to_device
        ],
        key=lambda n: n[1],
        reverse=True,
    )
    most_constrained = all_non_mapped_neighbors[0][0]

    all_mapped_neighbors = {
        neighbor
        for mapped_node in map_query_to_device.values()
        for neighbor in device_graph.neighbors(mapped_node)
    }

    for mapped_neighbor in all_mapped_neighbors:
        if check_mapping(
            most_constrained,
            mapped_neighbor,
            query,
            device_graph,
            map_query_to_device,
        ):
            return isomporphic_extension(
                dict(map_query_to_device) | {most_constrained: mapped_neighbor},
                query,
                device_graph,
            )

    return map_query_to_device


def find_subgraph_instances(query: PyDiGraph, device_graph: PyDiGraph):
    device_nodes = sorted(
        device_graph.node_indices(),
        key=lambda i: degree_and_neighbors_degrees(device_graph, i),
    )
    res: list[dict[int, int] | None] = []
    for device_node in device_nodes:
        for motif_node in query.node_indices():
            if can_support(device_graph, device_node, query, motif_node):
                sol = isomporphic_extension(
                    {motif_node: device_node}, query, device_graph
                )
                res.append(sol)
    return res


def main() -> None:
    print("hello")


if __name__ == "__main__":
    main()


##############################################3


def test_simple() -> None:
    triangle = PyDiGraph()
    triangle.add_nodes_from([0, 1, 2])
    triangle.add_edges_from([(0, 1, None), (1, 2, None), (2, 0, None)])

    line = PyDiGraph()
    line.add_nodes_from([0, 1])
    line.add_edges_from([(0, 1, None)])

    res = find_subgraph_instances(line, triangle)
    print(res)
    assert res
