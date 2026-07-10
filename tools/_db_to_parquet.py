# -*- coding: utf-8 -*-
"""Export the cleaned result DBs' `agent_results` table to Parquet (zstd).

Chunked + explicit dtypes so the schema is consistent and memory stays low.
Outputs alongside the .db files in results/hf_staging/.
"""
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

root = Path(__file__).resolve().parent.parent
stage = root / "results" / "hf_staging"

INT_COLS = ["id", "claim_id", "is_correct", "total_tokens", "prompt_tokens", "completion_tokens"]
STR_COLS = ["agent_name", "benchmark_name", "original_label", "model_label",
            "raw_output", "model_name", "created_at"]

def convert(db_path: Path, out_path: Path):
    print(f"\n=== {out_path.name} ===")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    writer = None
    total = 0
    for chunk in pd.read_sql_query("SELECT * FROM agent_results", con, chunksize=20000):
        for c in INT_COLS:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int64")
        chunk["time_thought"] = pd.to_numeric(chunk["time_thought"], errors="coerce").astype("float64")
        for c in STR_COLS:
            if c in chunk.columns:
                chunk[c] = chunk[c].astype("string")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(chunk)
        print(f"  ...{total:,} rows")
    writer.close()
    con.close()
    print(f"  -> {out_path.name}: {out_path.stat().st_size/1e6:,.0f} MB ({total:,} rows)")

convert(stage / "subsample_analyzed.db", stage / "subsample_analyzed.parquet")
convert(stage / "full_benchmark.db", stage / "full_benchmark.parquet")
print("\nDONE")
