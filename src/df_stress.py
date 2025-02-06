import logging
import os

os.environ["POLARS_MAX_THREADS"] = "5"
import polars as pl
from polars import DataFrame
import numpy as np

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


def load_large_df2(data: DataFrame) -> pl.DataFrame:
    def nums_to_bmp(nums):
        res = 0
        for num in nums:
            res = res | (1 << num)
        return res

    df = data
    df = df.with_columns(
        pl.col("indices").map_elements(nums_to_bmp, return_dtype=pl.UInt32)
    )
    return df


def queries(db: DataFrame, queries: list[str]) -> list[set[int]]:
    nodes_so_far = db.filter(pl.col("name") == queries[0])["indices"].to_list()[0]
    for q in queries[1:]:
        found_nodes = db.filter(
            (pl.col("name") == q)
            & (
                pl.col("indices").list.set_intersection(list(nodes_so_far)).list.len()
                == 1
            )
        )
        nodes_so_far = set(found_nodes["indices"].to_list()[0]).union(nodes_so_far)
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


if __name__ == "__main__":
    logger.info("Loading large dataframe...")
    data = gen_rows(1_000_000)
    df = load_large_df(data)
    logger.info("Loading large dataframe...")
    df2 = load_large_df2(df)

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

    import time

    logger.info("Generating queries...")
    s = time.perf_counter()
    queries2(df2, search)
    logger.info(f"Took {time.perf_counter() - s} s.")
    logger.info("Generating queries...")
    s = time.perf_counter()
    queries(df, search)
    logger.info(f"Finished {time.perf_counter() - s} s.")
