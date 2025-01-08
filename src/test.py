from rustworkx import PyDiGraph
from motif_search import find_subgraph_instances


def test_empty_graphs():
    """Test behavior with empty graphs"""
    empty_query = PyDiGraph()
    empty_device = PyDiGraph()
    device = PyDiGraph()
    device.add_nodes_from([0, 1])
    device.add_edges_from([(0, 1, None)])

    # Empty query on empty device should return empty list
    assert find_subgraph_instances(empty_query, empty_device) == []

    # Empty query on non-empty device should return empty list
    assert find_subgraph_instances(empty_query, device) == []


def test_single_node():
    """Test matching single node graphs"""
    query = PyDiGraph()
    query.add_node(0)

    device = PyDiGraph()
    device.add_node(0)

    res = find_subgraph_instances(query, device)
    assert len(res) == 1
    assert res[0] == {0: 0}


def test_simple_path():
    """Test matching simple path patterns"""
    # Create a path of length 2 as query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2])
    query.add_edges_from([(0, 1, None), (1, 2, None)])

    # Create a path of length 3 as device graph
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3])
    device.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None)])

    res = find_subgraph_instances(query, device)
    print(res)
    assert len(res) == 2  # Should find 2 instances
    expected = [{0: 0, 1: 1, 2: 2}, {0: 1, 1: 2, 2: 3}]
    assert all(r in expected for r in res)


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


def test_cycle_matching():
    """Test matching cycle patterns"""
    # Create a 3-node cycle as query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2])
    query.add_edges_from([(0, 1, None), (1, 2, None), (2, 0, None)])

    # Create a 4-node cycle as device
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3])
    device.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None), (3, 0, None)])

    # Should not find any 3-cycle in a 4-cycle
    res = find_subgraph_instances(query, device)
    assert len(res) == 0


def test_complex_pattern():
    """Test matching a more complex pattern"""
    # Create a diamond pattern query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2, 3])
    query.add_edges_from([(0, 1, None), (0, 2, None), (1, 3, None), (2, 3, None)])

    # Create device graph with multiple diamonds
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3, 4, 5])
    device.add_edges_from(
        [
            (0, 1, None),
            (0, 2, None),
            (1, 3, None),
            (2, 3, None),
            (3, 4, None),
            (3, 5, None),
            (4, 0, None),
            (5, 0, None),
        ]
    )

    res = find_subgraph_instances(query, device)
    assert len(res) > 0  # Should find at least one diamond


def test_directed_edges():
    """Test matching with directed edges"""
    # Create directed path query
    query = PyDiGraph()
    query.add_nodes_from([0, 1])
    query.add_edge(0, 1, None)

    # Create device with opposite direction
    device = PyDiGraph()
    device.add_nodes_from([0, 1])
    device.add_edge(1, 0, None)

    # Should not match due to direction
    res = find_subgraph_instances(query, device)
    assert len(res) == 0


def test_disconnected_components():
    """Test matching with disconnected components"""
    # Create query with two disconnected nodes
    query = PyDiGraph()
    query.add_nodes_from([0, 1])

    # Create connected device
    device = PyDiGraph()
    device.add_nodes_from([0, 1])
    device.add_edge(0, 1, None)

    res = find_subgraph_instances(query, device)
    assert len(res) > 0  # Should still find matches


def test_degree_constraints():
    """Test degree-based constraints"""
    # Create star query (center node with 2 leaves)
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2])
    query.add_edges_from([(0, 1, None), (0, 2, None)])

    # Create device with different degree distribution
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3])
    device.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None)])

    res = find_subgraph_instances(query, device)
    assert len(res) == 0  # Should not match due to degree constraints


def test_complete_isomorphisms():
    """Test finding all isomorphisms in a simple triangle case"""
    # Create triangle query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2])
    query.add_edges_from([
        (0, 1, None),
        (1, 2, None),
        (2, 0, None)
    ])

    # Create triangle device - should find all 6 possible mappings
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2])
    device.add_edges_from([
        (0, 1, None),
        (1, 2, None),
        (2, 0, None)
    ])

    res = find_subgraph_instances(query, device)
    # For a triangle, there should be 6 possible isomorphisms
    expected = [
        {0: 0, 1: 1, 2: 2},
        {0: 1, 1: 2, 2: 0},
        {0: 2, 1: 0, 2: 1},
        {0: 0, 1: 2, 2: 1},
        {0: 1, 1: 0, 2: 2},
        {0: 2, 1: 1, 2: 0}
    ]
    assert len(res) == 6
    assert all(mapping in expected for mapping in res)
    assert all(expected_mapping in res for expected_mapping in expected)


def test_square_with_diagonal():
    """Test finding all isomorphisms in a square with one diagonal"""
    # Create a square with one diagonal as query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2, 3])
    query.add_edges_from([
        (0, 1, None),
        (1, 2, None),
        (2, 3, None),
        (3, 0, None),
        (0, 2, None)  # Diagonal
    ])

    # Create identical device graph
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3])
    device.add_edges_from([
        (0, 1, None),
        (1, 2, None),
        (2, 3, None),
        (3, 0, None),
        (0, 2, None)  # Diagonal
    ])

    res = find_subgraph_instances(query, device)
    # Should find 8 isomorphisms (4 rotations * 2 flips)
    assert len(res) == 8


def test_path_in_larger_graph():
    """Test finding all possible path embeddings in a larger graph"""
    # Create a simple path query of length 2
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2])
    query.add_edges_from([
        (0, 1, None),
        (1, 2, None)
    ])

    # Create a complex device graph
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3, 4])
    device.add_edges_from([
        (0, 1, None),
        (1, 2, None),
        (2, 3, None),
        (3, 4, None),
        (4, 0, None),  # Make it a cycle
        (0, 2, None),  # Add some diagonals
        (0, 3, None)
    ])

    res = find_subgraph_instances(query, device)
    # Should find multiple path instances
    assert len(res) >= 7  # At least 7 different path embeddings possible


def test_bowtie_graph():
    """Test finding all isomorphisms in a bowtie graph (two triangles sharing a vertex)"""
    # Create bowtie query
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2, 3, 4])
    query.add_edges_from([
        (0, 1, None), (1, 2, None), (2, 0, None),  # First triangle
        (2, 3, None), (3, 4, None), (4, 2, None)  # Second triangle
    ])

    # Create identical device graph
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3, 4])
    device.add_edges_from([
        (0, 1, None), (1, 2, None), (2, 0, None),  # First triangle
        (2, 3, None), (3, 4, None), (4, 2, None)  # Second triangle
    ])

    res = find_subgraph_instances(query, device)
    # Should find 12 isomorphisms (6 ways to map first triangle * 2 ways to map second triangle)
    assert len(res) == 12


def test_star_in_clique():
    """Test finding all star patterns in a complete graph"""
    # Create star query with 3 points
    query = PyDiGraph()
    query.add_nodes_from([0, 1, 2, 3])
    query.add_edges_from([
        (0, 1, None),
        (0, 2, None),
        (0, 3, None)
    ])

    # Create complete graph K5 as device
    device = PyDiGraph()
    device.add_nodes_from([0, 1, 2, 3, 4])
    device.add_edges_from([
        (i, j, None)
        for i in range(5)
        for j in range(i + 1, 5)
    ])

    res = find_subgraph_instances(query, device)
    # In K5, each vertex can be center of star, and remaining vertices can be arranged in 6 ways
    # So total number of star patterns should be 5 * 6 = 30
    assert len(res) == 30
