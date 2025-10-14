#!/usr/bin/env python3
"""
Filter peptide results by target mass windows and label with sequence.

Inputs:
  - Search list Excel with columns:
        'Site', 'Seq Loc', 'Mass', 'Sequence',
        'Acceptable range', 'Unnamed: 5'
    where 'Acceptable range' and 'Unnamed: 5' are the lower/upper mass bounds.
  - Results data Excel with a 'Mass' column (plus anything else).

Output:
  - Excel file with:
        Sheet 'Filtered Matches': all rows from the Results that fall inside
          any peptide's range, annotated with the peptide Sequence, Site,
          Seq Loc, target mass, and bounds.
        Sheet 'Summary': counts per target mass/sequence.

Usage:
    python filter_peptides_by_mass.py "Search List File Harry.xlsx" \
           "Results Data file Harry.xlsx" "filtered_results.xlsx"
"""

import sys
import pandas as pd
from pathlib import Path

def main(search_path, results_path, out_path):
    search_df = pd.read_excel(search_path)
    results_df = pd.read_excel(results_path)

    # Coerce numeric to avoid string issues
    for col in ["Mass", "Acceptable range", "Unnamed: 5"]:
        if col in search_df.columns:
            search_df[col] = pd.to_numeric(search_df[col], errors="coerce")
    if "Mass" in results_df.columns:
        results_df["Mass"] = pd.to_numeric(results_df["Mass"], errors="coerce")

    matches = []
    for _, row in search_df.iterrows():
        lower = row["Acceptable range"]
        upper = row["Unnamed: 5"]
        tgt_mass = row["Mass"]
        if pd.isna(lower) or pd.isna(upper) or pd.isna(tgt_mass):
            continue
        seq = row["Sequence"]
        site = row["Site"]
        seq_loc = row["Seq Loc"]

        mask = results_df["Mass"].between(lower, upper, inclusive="both")
        subset = results_df.loc[mask].copy()
        if subset.empty:
            continue

        subset.insert(0, "Matched Sequence", seq)
        subset.insert(1, "Matched Seq Loc", seq_loc)
        subset.insert(2, "Matched Site", site)
        subset.insert(3, "Target Mass (Search)", tgt_mass)
        subset.insert(4, "Lower Bound", lower)
        subset.insert(5, "Upper Bound", upper)
        subset["Delta to Target (Da)"] = subset["Mass"] - tgt_mass
        matches.append(subset)

    if matches:
        out_df = pd.concat(matches, ignore_index=True)
        sort_cols = [c for c in ["Target Mass (Search)", "Mass", "File"]
                     if c in out_df.columns]
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=["Matched Sequence","Matched Seq Loc",
                                       "Matched Site","Target Mass (Search)",
                                       "Lower Bound","Upper Bound"] +
                                       results_df.columns.tolist())

    summary = (
        out_df.groupby(["Target Mass (Search)", "Matched Sequence",
                        "Matched Seq Loc", "Matched Site"], dropna=False)
              .agg(hits=("Mass","count"),
                   min_obs_mass=("Mass","min"),
                   max_obs_mass=("Mass","max"))
              .reset_index()
              .sort_values(["Target Mass (Search)"])
    )

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Filtered Matches")
        summary.to_excel(writer, index=False, sheet_name="Summary")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python filter_peptides_by_mass.py "
              "<search_list.xlsx> <results_data.xlsx> <output.xlsx>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
