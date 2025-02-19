import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("RB_dataset_analysis.csv")

# Exclude "Archive" method
df = df[df["Method"] != "Archive"]

# Define method order
method_order = [
    "Expert_config", "Original", "DeepQ", "TabularQ",
    "DeepQ_r2_from_DeepQ", "DeepQ_r2_from_TabularQ",
    "TabularQ_r2_from_DeepQ", "TabularQ_r2_from_TabularQ",
    "Bellman_r3_DeepQ_v2", "Bellman_r3_large_net_v2"
]

# Filter only desired methods
df = df[df["Method"].isin(method_order)]

# Specify user combinations to include
selected_users = {
    "_0_1_2", "_0_2_2", "_1_1_2", "_1_1_4",
    "_1_2_1", "_1_2_3", "_1_2_5", "_1_3_4", "_3_2_3"
}

# Filter dataset for selected user combinations
df_filtered = df[df["User Combination"].isin(selected_users)]

# Compute mean of means
avg_means = df_filtered.groupby("Method")["Mean"].mean().to_dict()

# Compute pooled standard deviation
corrected_std = {}
for method in method_order:
    method_df = df_filtered[df_filtered["Method"] == method]

    if method_df.empty:
        corrected_std[method] = np.nan
        continue

    # Extract necessary fields
    user_means = method_df.groupby("User Combination")["Mean"].mean()
    user_stds = method_df.groupby("User Combination")["Std Dev"].mean()
    user_trials = method_df.groupby("User Combination")["Trials"].sum()

    # Compute within-user variance contribution
    within_user_variance = np.nansum((user_stds ** 2) / user_trials)

    # Compute between-user variance contribution
    overall_mean = user_means.mean()
    between_user_variance = np.nansum((user_means - overall_mean) ** 2) / (len(user_means) - 1) if len(user_means) > 1 else 0

    # Compute final standard deviation
    corrected_variance = (within_user_variance / len(user_means)) + between_user_variance
    corrected_std[method] = np.sqrt(corrected_variance) if corrected_variance >= 0 else 0  # Avoid sqrt of negative

# Convert results to DataFrames
results_mean_df = pd.DataFrame([avg_means], index=["Mean"]).reindex(columns=method_order, fill_value=np.nan)
results_corrected_std_df = pd.DataFrame([corrected_std], index=["Corrected Std"]).reindex(columns=method_order, fill_value=np.nan)

# Save results
mean_output_path = "RB_method_avg_means_filtered.csv"
std_output_path = "RB_method_corrected_std_filtered.csv"

results_mean_df.to_csv(mean_output_path)
results_corrected_std_df.to_csv(std_output_path)

print(f"Mean analysis saved to: {mean_output_path}")
print(f"Corrected standard deviation saved to: {std_output_path}")
