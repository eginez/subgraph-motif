import polars as pl
from pandas_motifs import RAW_DATA


def create_db(raw_data: dict) -> pl.DataFrame:
    df = pl.DataFrame(raw_data)
    df = df.with_columns([pl.col("indices").cast(pl.List(pl.Int8))])
    return df


def find_motifs(motif_db: pl.DataFrame, motifs: list[str]) -> list[set[int]]:
    if len(motifs) == 0:
        return None
    first_motif = motifs[0]

    first_nodes = df.filter(pl.col("motif") == first_motif)["indices"].to_list()

    res: list[set[int]] = []
    for nodes_so_far in first_nodes:
        one_res = find_motifs_helper(motif_db, motifs[1:], nodes_so_far)
        res.extend(one_res)
    return res


def find_motifs_helper(
    motif_db: pl.DataFrame, motifs: list[str], nodes_so_far: list[int]
) -> list[set[int]]:

    if len(motifs) == 0:
        return [nodes_so_far]

    next_motif = motifs[0]

    found_nodes = motif_db.filter(
        (pl.col("motif") == next_motif)
        & (pl.col("indices").list.set_intersection(nodes_so_far).list.len() == 1)
    )
    all_res: list[set[int]] = []
    for all_found_nodes in found_nodes["indices"].to_list():
        new_nodes_so_far = set(nodes_so_far).union(all_found_nodes)
        new_res = find_motifs_helper(
            motif_db, motifs[1:], nodes_so_far=list(new_nodes_so_far)
        )
        all_res.extend(new_res)

    return all_res


if __name__ == "__main__":
    df = create_db(RAW_DATA)
    nn = find_motifs(df, ["T", "3line"])
    print(nn)
