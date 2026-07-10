# -*- coding: utf-8 -*-
"""Build cleaned copies for the HF release:
   - drop discontinued agents (uam_ga_web_tool_arch, uam_ga_rag_decomp_arch) from both
   - drop the `claims` table (question text) from the subsample (keep claim_id)
"""
import shutil, sqlite3, os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
stage = root / "results" / "hf_staging"
stage.mkdir(parents=True, exist_ok=True)

jobs = [
    (root / "results/results_am_benchmark.db", stage / "full_benchmark.db", False),
    (root / "results/results_am_subsample.db", stage / "subsample_analyzed.db", True),
]

for src, dst, drop_claims in jobs:
    print(f"\n=== {dst.name} (from {src.name}) ===")
    if dst.exists():
        dst.unlink()
    print(f"  copying {src.stat().st_size/1e6:,.0f} MB ...")
    shutil.copy2(src, dst)
    con = sqlite3.connect(dst)
    for arch in ("uam_ga_web_tool_arch", "uam_ga_rag_decomp_arch"):
        n = con.execute("SELECT COUNT(*) FROM agent_results "
                        "WHERE agent_name LIKE ?", (arch + "%",)).fetchone()[0]
        con.execute("DELETE FROM agent_results WHERE agent_name LIKE ?", (arch + "%",))
        print(f"  deleted {n:,} {arch} rows")
    if drop_claims:
        has = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='claims'").fetchone()
        if has:
            con.execute("DROP TABLE claims")
            print("  dropped `claims` table (question text removed)")
    con.commit()
    con.execute("VACUUM")
    con.commit()
    con.close()
    print(f"  -> {dst.stat().st_size/1e6:,.0f} MB")

# ---- verification ----
print("\n=== VERIFY ===")
for _, dst, _ in jobs:
    con = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    agents = sorted({r[0].split("__")[0] for r in con.execute("SELECT DISTINCT agent_name FROM agent_results")})
    arch = con.execute("SELECT COUNT(*) FROM agent_results WHERE agent_name LIKE 'uam_ga_web_tool_arch%' "
                       "OR agent_name LIKE 'uam_ga_rag_decomp_arch%'").fetchone()[0]
    cmin, cmax, cdist = con.execute("SELECT MIN(claim_id), MAX(claim_id), COUNT(DISTINCT claim_id) FROM agent_results").fetchone()
    print(f"{dst.name}: tables={tabs} | agents={agents} | arch_rows={arch} | "
          f"claim_id range={cmin}..{cmax} distinct={cdist:,}")
    con.close()
print("\nOK")
