import pandas as pd

from rdagent_china.utils.io import DuckDBIO, DuckDBConfig


def test_duckdb_round_trip_in_memory():
    conn = DuckDBIO.connect(DuckDBConfig(path=":memory:"))
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.5, 2.5, 3.5],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    DuckDBIO.write_df(conn, df, "t", mode="overwrite")
    out = DuckDBIO.read_df(conn, "t")

    # order by id to ensure deterministic comparison
    out = out.sort_values("id").reset_index(drop=True)
    assert len(out) == 3
    assert out.loc[0, "name"] == "a"
    assert pd.to_datetime(out.loc[2, "ts"]).strftime("%Y-%m-%d") == "2024-01-03"
