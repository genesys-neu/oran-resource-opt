import pandas as pd
from collections import defaultdict

# Load the results DataFrame
df = pd.read_csv("RB_dataset_analysis.csv")

# Exclude the "Archive" method
df = df[df["Method"] != "Archive"]

# Define the desired method order
method_order = [
    "Expert_config", "Original", "DeepQ", "TabularQ",
    "DeepQ_r2_from_DeepQ", "DeepQ_r2_from_TabularQ",
    "TabularQ_r2_from_DeepQ", "TabularQ_r2_from_TabularQ",
    "Bellman_r3_DeepQ_v2", "Bellman_r3_large_net_v2"
]

# Filter DataFrame to only include the methods in method_order
df = df[df["Method"].isin(method_order)]

# Track the methods associated with each user combination
user_method_counts = defaultdict(set)
for _, row in df.iterrows():
    user_method_counts[row["User Combination"]].add(row["Method"])

# Compute the maximum number of missing methods
methods = method_order  # Now safe to assume all methods exist
# methods = sorted(df["Method"].unique())  # Extracts all unique methods and sorts them

user_grouped = {i: set() for i in range(len(methods) + 1)}  # Ensure valid range

# Group user combinations by how many methods they are missing
for user_comb, present_methods in user_method_counts.items():
    missing_count = len(methods) - len(present_methods)
    user_grouped[missing_count].add(user_comb)

# Save user combinations for each `max_missing` value
user_comb_output_path = "RB_user_combinations_by_missing.csv"
with open(user_comb_output_path, "w") as f:
    f.write("Max Missing Methods,User Combination\n")
    for max_missing, user_combs in user_grouped.items():
        for user_comb in user_combs:
            f.write(f"{max_missing},{user_comb}\n")

# Initialize storage for raw data
results_mean = {}
results_weighted = {}
raw_mean_data = {}
raw_weighted_data = {}
raw_weighted_filtered = {}

# Filtering threshold for coefficient of variation (CV)
cv_threshold = 0.5

for max_missing in range(len(methods)):
    included_users = set()
    for i in range(max_missing + 1):  # Include user combinations missing up to `max_missing` methods
        included_users.update(user_grouped.get(i, set()))

    df_filtered = df[df["User Combination"].isin(included_users)]

    # Compute average (non-weighted) mean
    avg_means = df_filtered.groupby("Method")["Mean"].mean().to_dict()
    results_mean[max_missing] = avg_means

    # Compute weighted score mean
    avg_weighted = df_filtered.groupby("Method")["Weighted Score"].mean().to_dict()
    results_weighted[max_missing] = avg_weighted

    # Store raw mean and weighted scores per user combination
    raw_mean_data[max_missing] = df_filtered.pivot(index="User Combination", columns="Method", values="Mean")
    raw_weighted_data[max_missing] = df_filtered.pivot(index="User Combination", columns="Method", values="Weighted Score")

    # Filter out weighted scores where CV is missing or exceeds the threshold
    df_filtered_valid_cv = df_filtered[(df_filtered["CV"].notna()) & (df_filtered["CV"] <= cv_threshold)]
    raw_weighted_filtered[max_missing] = df_filtered_valid_cv.pivot(index="User Combination", columns="Method", values="Weighted Score")

# Convert results to DataFrames and sort columns
results_mean_df = pd.DataFrame.from_dict(results_mean, orient="index").reindex(columns=method_order, fill_value=float("nan"))
results_weighted_df = pd.DataFrame.from_dict(results_weighted, orient="index").reindex(columns=method_order, fill_value=float("nan"))

# Set index name
results_mean_df.index.name = "Max Missing Methods"
results_weighted_df.index.name = "Max Missing Methods"

# Save results
mean_output_path = "RB_method_avg_means.csv"
weighted_output_path = "RB_method_avg_weighted_scores.csv"
raw_mean_output_path = "RB_raw_mean_data.csv"
raw_weighted_output_path = "RB_raw_weighted_data.csv"
raw_weighted_filtered_output_path = "RB_raw_weighted_filtered.csv"

results_mean_df.to_csv(mean_output_path)
results_weighted_df.to_csv(weighted_output_path)

# Concatenate all raw data across max_missing values
raw_mean_df = pd.concat(raw_mean_data, names=["Max Missing Methods"]).reindex(columns=method_order, fill_value=float("nan"))
raw_weighted_df = pd.concat(raw_weighted_data, names=["Max Missing Methods"]).reindex(columns=method_order, fill_value=float("nan"))
raw_weighted_filtered_df = pd.concat(raw_weighted_filtered, names=["Max Missing Methods"]).reindex(columns=method_order, fill_value=float("nan"))

raw_mean_df.to_csv(raw_mean_output_path)
raw_weighted_df.to_csv(raw_weighted_output_path)
raw_weighted_filtered_df.to_csv(raw_weighted_filtered_output_path)

print(f"Mean analysis saved to: {mean_output_path}")
print(f"Weighted score analysis saved to: {weighted_output_path}")
print(f"Raw mean data saved to: {raw_mean_output_path}")
print(f"Raw weighted data saved to: {raw_weighted_output_path}")
print(f"Filtered raw weighted data saved to: {raw_weighted_filtered_output_path}")
print(f"User combinations saved to: {user_comb_output_path}")