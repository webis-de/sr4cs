"""Final filtering and combining of SLR to reference mappings."""

import pandas as pd


df_map = pd.read_csv(
    "../../../data/extracted_information/refs/processed/slr2ref_map.csv"
)
df_refs = pd.read_parquet("../../../data/extracted_information/refs/final/refs.parquet")

# Step 1: filter to only keep rows where ref_id is in df_refs
df_filtered = df_map[df_map["ref_id"].isin(df_refs["ref_id"])]

# Step 2: group by id and collect ref_ids into a list
df_grouped = df_filtered.groupby("id")["ref_id"].apply(list).reset_index()

# add a column with the count of ref_ids
df_grouped["num_refs"] = df_grouped["ref_id"].apply(len)

# save to csv
df_grouped.to_csv(
    "../../../data/extracted_information/refs/final/filtered_slr2ref_map.csv",
    index=False,
)
