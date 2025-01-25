from typing import Iterable, Any

import pandas as pd


# def simple_query(df: pd.DataFrame):
#     df_cross = pd.DataFrame.merge(
#         df[["indices", "motif"]].reset_index(),
#         df[["indices", "motif"]].reset_index(),
#         how="cross",
#         suffixes=("_x", "_y"),
#     )
#     df_cross = df_cross[df_cross["index_x"] != df_cross["index_y"]]
#     df_cross["intersection"] = df_cross.apply(
#         lambda x: x["indices_x"] & x["indices_y"], axis=1
#     )
#     df_cross["intersection_len"] = df_cross["intersection"].apply(len)
#     print(df_cross)
#     # So now finding overlapping motifs becomes a queries into the dataframe
#     # For example say I want to find out all T motif and a 3line motif
#     t_and_line = df_cross[
#         (df_cross["motif_x"] == "T")
#         & (df_cross["motif_y"] == "3line")
#         & (df_cross["intersection_len"] == 1)
#     ]
#     print(t_and_line[["indices_x", "indices_y", "intersection"]])
#


def create_db(raw_data: Any) -> pd.DataFrame:
    # Is all this vectorized??
    df = pd.DataFrame(raw_data, dtype="object")
    df["indices"] = df["indices"].apply(set)
    return df


def find_motifs(motif_db: pd.DataFrame, motifs: list[str]) -> Any:
    if len(motifs) == 0:
        return None
    first_motif = motifs[0]

    first_nodes = motif_db[motif_db["motif"] == first_motif]["indices"].values

    res: list[set[int]] = []
    for nodes_so_far in first_nodes:
        one_res = find_motifs_helper(motif_db, motifs[1:], nodes_so_far)
        res.extend(one_res)
    return res


def find_motifs_helper(
    motif_db: pd.DataFrame, motifs: list[str], nodes_so_far: set[int]
) -> list[set[int]]:
    """
    Colum names in the dataframe are: indices, motif
    The point of this function is to return one set of node indices that match the motifs
    by querying the dataframe. As motifs are found they get merged into a bigger
    set that then executes intersection against the motif database.
    """

    if len(motifs) == 0:
        return [nodes_so_far]

    next_motif = motifs[0]

    # First create the query
    # Is this vectorized
    found_nodes = motif_db[
        (motif_db["motif"] == next_motif)
        &
        # TODO can there be in insertection outside the edges?
        (motif_db["indices"].apply(lambda x: len(x & nodes_so_far) == 1))
    ]

    all_res: list[set[int]] = []
    for all_found_nodes in found_nodes["indices"].values:
        new_nodes_so_far = nodes_so_far | all_found_nodes
        new_res = find_motifs_helper(
            motif_db, motifs[1:], nodes_so_far=new_nodes_so_far
        )
        all_res.extend(new_res)

    return all_res


raw_data = {
    "indices": [
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [1, 3, 4],
        [3, 4, 5],
        [3, 4, 6],
        [4, 5, 6],
    ],
    "motif": [
        "T",
        "T",
        "3line",
        "3line",
        "3line",
        "3line",
        "3line",
        "3line",
        "3line",
    ],
}
if __name__ == "__main__":

    df = create_db(raw_data)
    # simple_query(df)
    nn = find_motifs(df, ["T", "3line"])
    print(nn)


def test_find_motifs() -> None:
    df = create_db(raw_data)
    nn = find_motifs(df, ["T", "3line"])
    expected = [
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 6},
        {0, 1, 3, 4, 5, 6},
        {1, 2, 3, 4, 5, 6},
    ]
    assert all(map(lambda x: x in nn, expected))
