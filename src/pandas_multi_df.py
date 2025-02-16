import pandas as pd
import numpy as np


def build_df(
    row_count: int, group: int, max_index: int = 10, max_elements: int = 10
) -> pd.DataFrame:
    data = {}
    for i in range(row_count):
        data[f"{i}"] = np.random.randint(0, max_index, size=(max_elements,))
    data["intr"] = np.random.randint(0, max_index, size=(max_elements,))
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


if __name__ == "__main__":
    np.random.seed(0)
    df1 = build_static_df_1()
    df2 = build_static_df_2()
    print(df1)
    print()
    print(df2)
    print()
    # Ideally here we only want the 2nd only of the merge only
    # because the other ones have intersections on other columns
    merged = df1.merge(df2, on="intr", how="inner")
    print(merged)
    print()
    x_df = merged.filter(regex="_x$|^2$").melt(ignore_index=False)
    y_df = merged.filter(regex="_y$").melt(ignore_index=False)

    # Merge x and y values to find matches
    matches = pd.merge(
        x_df, y_df, left_index=True, right_index=True, suffixes=("_x", "_y")
    )
    print(matches)
    print()

    # Count matches per row where values are equal
    match_counts = (matches["value_x"] == matches["value_y"]).groupby(level=0).sum()

    print(match_counts)

    print(merged[match_counts <= 1])
