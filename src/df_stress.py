import logging
import os

os.environ["POLARS_MAX_THREADS"] = "5"
import polars as pl
from polars import DataFrame, col
import numpy as np
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

seed = 100
np.random.seed(seed)


def gen_rows(count: int):
    width = 3
    return [
        np.random.choice(["a", "b", "c", "d", "e", "f", "g"], count),
        np.random.randint(0, 32, (count, width), dtype=np.int32),
    ]


def load_large_df(data) -> pl.DataFrame:
    df = DataFrame(data, schema=["name", "indices"])
    df = df.with_columns([pl.col("indices").cast(pl.List(pl.Int32))])
    return df


def load_large_df2(data) -> pl.DataFrame:
    def nums_to_bmp(nums):
        npnums = np.asarray(nums, dtype=np.uint32)
        return np.bitwise_or.reduce(1 << npnums)

    df = DataFrame(data, schema=["name", "indices"])
    df = df.with_columns([pl.col("indices").cast(pl.Array(pl.UInt32))])
    df = df.with_columns(
        pl.col("indices").map_elements(nums_to_bmp, return_dtype=pl.UInt32)
    )
    return df


def queries(db: DataFrame, queries: list[str], expect_position: int) -> list[set[int]]:
    nodes_so_far = set(db.filter(pl.col("name") == queries[0])["indices"].to_list()[0])
    for q in queries[1:]:
        found_nodes = db.with_columns(
            [
                col("indices")
                .list.eval(pl.element().is_in(nodes_so_far))
                .list.sum()
                .alias("count_int"),
                col("indices")
                .list.eval(pl.element().is_in(nodes_so_far))
                .list.arg_max()
                .alias("position"),
            ]
        ).filter(
            (col("name") == q)
            & (col("count_int") == 1)
            & (col("position") == expect_position)
        )

        if not found_nodes.is_empty():
            nodes_so_far |= set(found_nodes["indices"].first())

    return nodes_so_far


def queries2(db: DataFrame, queries: list[str]) -> list[set[int]]:
    nodes_so_far = db.filter(pl.col("name") == queries[0])["indices"].to_list()[0]
    for q in queries[1:]:
        found_nodes = db.filter(
            (pl.col("name") == q)
            & ((pl.col("indices") & nodes_so_far).bitwise_count_ones() == 1)
        )
        nodes_so_far = found_nodes["indices"].to_list()[0] | nodes_so_far
    return nodes_so_far


def bitmap_df(search_terms: list[str]) -> None:
    logger.info("Loading large dataframe...")
    data = gen_rows(1_000_000)
    df2 = load_large_df2(data)
    s = time.perf_counter()
    queries2(df2, search_terms)
    logger.info(f"Took {time.perf_counter() - s} s.")


def list_df(search_terms: list[str], expect_position: int) -> None:
    logger.info("Loading large dataframe...")
    data = gen_rows(1_000_000)
    df = load_large_df(data)
    logger.info("Generating queries...")
    s = time.perf_counter()
    queries(df, search_terms, expect_position)
    logger.info(f"Finished {time.perf_counter() - s} s.")


if __name__ == "__main__":

    search = [
        "a",
        "b",
        "c",
        "a",
        "e",
        "f",
        "g",
        "g",
        "f",
        "c",
        "b",
        "a",
        "d",
        "f",
        "e",
        "d",
        "d",
        "c",
        "b",
        "a",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
        "c",
        "b",
        "a",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
    ]

    list_df(search, 1)

    # bitmap_df(search)
