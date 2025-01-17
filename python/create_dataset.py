import os
import pandas as pd
import numpy as np


# Specify the columns to keep
COLUMNS_TO_KEEP = [
    'dl_mcs', 'dl_n_samples', 'dl_buffer [bytes]', 'tx_brate downlink [Mbps]',
    'tx_pkts downlink', 'dl_cqi', 'ul_mcs',
    'ul_n_samples', 'ul_buffer [bytes]', 'rx_brate uplink [Mbps]',
    'rx_pkts uplink', 'rx_errors uplink (%)', 'ul_sinr', 'phr',
    'sum_requested_prbs', 'sum_granted_prbs', 'ul_turbo_iters'
]


def clean_csv(file_path, experiment_id):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Strip spaces and clean columns
    df.columns = df.columns.str.strip()
    cleaned_columns_to_keep = [col.strip() for col in COLUMNS_TO_KEEP]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    columns_to_keep_existing = [col for col in cleaned_columns_to_keep if col in df.columns]
    df = df[columns_to_keep_existing + ['slice_id']]
    df = df.rename(columns={'slice_id': 'Label'})

    # Ensure that all columns are numeric
    non_numeric_cols = df[columns_to_keep_existing].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Skipping file {file_path} due to non-numeric data in columns: {', '.join(non_numeric_cols)}")
        return None  # Return None to skip this file

    # Check for `inf` values in the DataFrame
    elif np.isinf(df.values).any():
        print(f"Skipping file {file_path} due to presence of 'inf' values.")
        return None  # Return None to skip this file

    else:
        # Add the Experiment ID
        df['ExperimentID'] = experiment_id

        # Clip the specific columns to their max values
        if 'tx_brate downlink [Mbps]' in df.columns:
            df['tx_brate downlink [Mbps]'] = df['tx_brate downlink [Mbps]'].clip(upper=20)
        if 'rx_brate uplink [Mbps]' in df.columns:
            df['rx_brate uplink [Mbps]'] = df['rx_brate uplink [Mbps]'].clip(upper=15)

        return df


def read_directories(txt_file):
    # Read directories from the .txt file
    with open(txt_file, 'r') as file:
        directories = [line.strip() for line in file.readlines() if line.strip()]
    return directories


def find_csv_files(directories):
    csv_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    return csv_files


def main(directories_file, output_file, stats_file):
    directories = read_directories(directories_file)
    csv_files = find_csv_files(directories)

    chunk_list = []
    global_min = None
    global_max = None
    experiment_id = 0

    # Remove the output file if it already exists (start fresh)
    if os.path.exists(output_file):
        os.remove(output_file)

    for idx, file_path in enumerate(csv_files):
        cleaned_df = clean_csv(file_path, experiment_id)

        if cleaned_df is None:  # Skip the current file if it contains 'inf' values
            continue

        experiment_id += 1

        # Calculate min/max for the current chunk
        numeric_df = cleaned_df.drop(columns=['Label', 'ExperimentID'], errors='ignore').select_dtypes(
            include=['number'])
        if global_min is None and global_max is None:
            # Initialize global min/max with the first chunk's min/max
            global_min = numeric_df.min()
            global_max = numeric_df.max()
        else:
            # Update global min/max for each feature
            global_min = pd.concat([global_min, numeric_df.min()], axis=1).min(axis=1)
            global_max = pd.concat([global_max, numeric_df.max()], axis=1).max(axis=1)

        chunk_list.append(cleaned_df)

        # Write periodically to the output file
        if (idx + 1) % 10 == 0 or idx == len(csv_files) - 1:
            # Filter out empty DataFrames before concatenation
            non_empty_chunks = [chunk for chunk in chunk_list if not chunk.empty]
            if non_empty_chunks:  # Proceed only if there are valid chunks
                combined_df = pd.concat(non_empty_chunks, ignore_index=True)

                # Write headers only for the first batch
                write_mode = 'w' if idx < 10 else 'a'
                combined_df.to_csv(output_file, mode=write_mode, header=(write_mode == 'w'), index=False)

                print(f"Processed {idx + 1}/{len(csv_files)} files and saved to {output_file}")

            chunk_list.clear()

    # Save stats to the stats file
    if global_min is not None and global_max is not None:
        stats = pd.DataFrame({'Feature': global_min.index, 'Min': global_min.values, 'Max': global_max.values})
        stats.to_csv(stats_file, index=False)
        print(f"Min and Max values saved to {stats_file}")

    print(f"Combined CSV saved to {output_file}")


# Example usage
if __name__ == "__main__":
    directories_file = 'directories.txt'  # Replace with your .txt file path
    output_file = 'combined_metrics.csv'
    stats_file = 'feature_stats.csv'
    main(directories_file, output_file, stats_file)
