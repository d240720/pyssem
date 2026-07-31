#!/usr/bin/env python3
"""
export_intact_satnos.py

Write the NORAD IDs (SATNOs) of payloads + rocket bodies to a CSV, for the DISCOS
puller to fetch physical mass/size. Reads the raw Space-Track catalog (2026.csv) so
it uses true NORAD_CAT_IDs and the clean OBJECT_TYPE string, avoiding any objects
the IC build dropped. Debris are intentionally excluded.

Output is headerless (one ID per line), matching MATLAB's writematrix so it stays
drop-in for the existing DISCOS puller. Use --header to emit a 'satno' header row.

USAGE: python3 export_intact_satnos.py [--gp 2026.csv] [--out intact_satnos.csv]
           [--max-norad N] [--header]
"""

import argparse

import pandas as pd

TYPES = ("PAYLOAD", "ROCKET BODY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="2026.csv")
    ap.add_argument("--out", default="intact_satnos.csv")
    ap.add_argument("--max-norad", type=int, default=None,
                    help="drop IDs above this. Default: keep everything.")
    ap.add_argument("--header", action="store_true",
                    help="emit a 'satno' header row (default: headerless, as MATLAB)")
    a = ap.parse_args()

    gp = pd.read_csv(a.gp, usecols=["NORAD_CAT_ID", "OBJECT_TYPE"])

    otype = gp["OBJECT_TYPE"].astype("string").str.strip().str.upper()
    intact = otype.isin(TYPES)

    satnos = pd.to_numeric(gp.loc[intact, "NORAD_CAT_ID"], errors="coerce")
    satnos = satnos.dropna().round().astype("int64")
    satnos = satnos[satnos > 0].drop_duplicates().sort_values()

    n_all = len(satnos)
    if a.max_norad is not None:
        satnos = satnos[satnos <= a.max_norad]

    satnos.rename("satno").to_csv(a.out, index=False, header=a.header)

    n_pay = int((otype[intact] == "PAYLOAD").sum())
    n_rb = int((otype[intact] == "ROCKET BODY").sum())
    print(f"Wrote {len(satnos)} unique SATNOs to {a.out} "
          f"(from {n_pay} payload + {n_rb} rocket-body rows)")
    if invalid := int(intact.sum()) - n_all:
        print(f"  {invalid} row(s) had blank/invalid/duplicate IDs")
    if a.max_norad is not None and n_all != len(satnos):
        print(f"  dropped {n_all - len(satnos)} IDs above --max-norad {a.max_norad}")
    if len(satnos):
        print(f"  ID range: {satnos.min()} to {satnos.max()}")


if __name__ == "__main__":
    main()