import numpy as np
import pandas as pd


def vectorized_function(df: pd.DataFrame) -> pd.DataFrame:
    return df["a"] + df["b"]



if __name__ == "__main__":
    # Create a table with 4 columns with float values
    # from 0 to 1 random

    machine_sizes = (1_000_000_000,)
    circuit_sizes = [m // 2 for m in machine_sizes]
    dfs = {
        count: pd.DataFrame(np.random.rand(count, 4), columns=list("ABCD"))
        for count in machine_sizes
    }

    layouts = [np.random.randint(0, machine_size, circuit_size)
              for machine_size, circuit_size in zip(machine_sizes, circuit_sizes)]


    res = {
        len(layout): df.iloc[layout].prod(axis=1).prod()
        for layout, df in zip(layouts, dfs.values())
    }
    print(res)



