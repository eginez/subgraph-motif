from typing import Optional

import pandas as pd
import numpy as np
import polars as pl


def build_df(
        row_count: int, group: int, max_index: int = 10, column_count: int = 10
) -> pd.DataFrame:
    data = {}
    for i in range(row_count):
        data[f"{i}"] = np.random.randint(
            0, max_index, size=(column_count,), dtype=np.uint8
        )
    data["intr"] = np.random.randint(0, max_index, size=(column_count,), dtype=np.uint8)
    return pd.DataFrame(data)


def build_static_df_1() -> pd.DataFrame:
    data = {
        "0": [0, 1, 2, 3, 4],
        "1": [2, 0, 0, 0, 0],
        "2": [1, 3, 1, 1, 1],
        "intr": [1, 3, 2, 1, 4],
    }
    return pd.DataFrame(data)


def build_static_df_2() -> pd.DataFrame:
    data = {
        "0": [1, 1, 2, 3, 4],
        "1": [2, 0, 0, 2, 0],
        "intr": [1, 0, 0, 3, 0],
    }
    return pd.DataFrame(data)


import time
from contextlib import contextmanager


@contextmanager
def timeit_context(name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"--- {name} took {(end_time - start_time):.3f} s ---")


def operate_pandas(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Ideally here we only want the 2nd only of the merge only
    # because the other ones have intersections on other columns
    merged = df1.merge(df2, on="intr", how="inner")
    # print(merged)

    # print()
    x_df = merged.filter(regex="_x$|^2$").melt(ignore_index=False)
    y_df = merged.filter(regex="_y$").melt(ignore_index=False)

    # Merge x and y values to find matches
    matches = pd.merge(
        x_df, y_df, left_index=True, right_index=True, suffixes=("_x", "_y")
    )

    # Count matches per row where values are equal
    match_counts = (matches["value_x"] == matches["value_y"]).groupby(level=0).sum()

    # print(merged[match_counts <= 1])
    return merged[match_counts <= 1]


def operate_polars(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    # Perform inner join on "intr" column
    merged = df1.join(df2, on="intr", how="inner", suffix="_y")

    # Get column names ending with _x and column "2"
    x_cols = [col for col in merged.columns if not col.endswith("_y") and col != "intr"]
    y_cols = [col for col in merged.columns if col.endswith("_y")]

    # Create melted dataframes for x and y columns
    x_df = merged.select(x_cols).unpivot()
    y_df = merged.select(y_cols).unpivot()

    # Add row index for tracking original rows
    x_df = x_df.with_row_index("idx")
    y_df = y_df.with_row_index("idx")

    matches = x_df.join(y_df, on="idx", suffix="_y")

    match_counts = (
        matches.with_columns(pl.col("value").eq(pl.col("value_y")).alias("is_match"))
        .group_by("idx")
        .agg(pl.col("is_match").sum().alias("match_count"))
    )

    filtered_indices = match_counts.filter(pl.col("match_count") <= 1)["idx"]

    return merged.filter(pl.arange(0, len(merged)).is_in(filtered_indices))


def operate_pytorch(
        df1: pd.DataFrame, df2: pd.DataFrame, use_gpu: bool = True
) -> pd.DataFrame:
    """
    Filter merged dataframes using PyTorch tensor operations.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        use_gpu: Whether to use GPU or not

    Returns:
        Filtered DataFrame with rows having 1 or fewer value matches
    """
    import torch
    import platform

    # First merge on intr column using pandas
    merged = df1.merge(df2, on="intr", how="inner")
    if use_gpu:
        gpu_name = "mps" if platform.system() == "Darwin" else "cuda"
        device = (
            torch.device(gpu_name)
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    else:
        gpu_name = "cpu"
        device = torch.device("cpu")

    # Get x and y columns
    x_cols = [col for col in merged.columns if col.endswith("_x") or col == "2"]
    y_cols = [col for col in merged.columns if col.endswith("_y")]

    try:
        # Convert to tensors
        x_values = torch.tensor(
            merged[x_cols].values, dtype=torch.uint8, device=device
        )  # Shape: [n_rows, n_x_cols]
        y_values = torch.tensor(
            merged[y_cols].values, dtype=torch.uint8, device=device
        )  # Shape: [n_rows, n_y_cols]

        # Reshape for broadcasting
        x_expanded = x_values.unsqueeze(2)  # Shape: [n_rows, n_x_cols, 1]
        y_expanded = y_values.unsqueeze(1)  # Shape: [n_rows, 1, n_y_cols]
        if device.type == gpu_name:
            batch_size = 10_000
            row_count = len(merged)
            match_counts = torch.zeros(row_count, dtype=torch.uint8, device=device)

            for i in range(0, row_count, batch_size):
                end_index = min(i + batch_size, row_count)
                matches = x_expanded[i:end_index] == y_expanded[i:end_index]
                batch_counts = matches.sum(dim=(1, 2)).to(torch.uint8)
                match_counts[i:end_index] = batch_counts

            # Create boolean mask for filtering
            valid_rows = match_counts <= 1
            valid_rows = valid_rows.cpu()
        else:
            # Compare all x values with all y values
            # Result shape: [n_rows, n_x_cols, n_y_cols]
            matches = x_expanded == y_expanded

            # Count total matches per row
            match_counts = matches.sum(dim=(1, 2))  # Shape: [n_rows]
            valid_rows = match_counts <= 1
    finally:
        if device.type == gpu_name:
            torch.mps.empty_cache()

    # Apply filter to merged dataframe
    return merged[valid_rows.numpy()]


if __name__ == "__main__":
    np.random.seed(0)
    column_count = 500
    row_count = 100
    with timeit_context("creating dfs"):
        df1 = build_df(row_count, 0, 10, column_count + 3)
        df2 = build_df(row_count, 0, 10, 10)

    with timeit_context("operate_pytorch gpu"):
        res = operate_pytorch(df1, df2, use_gpu=True)

    with timeit_context("operate pandas"):
        res = operate_pandas(df1, df2)

    with timeit_context("operate polars"):
        res = operate_polars(pl.from_pandas(df1), pl.from_pandas(df2))
