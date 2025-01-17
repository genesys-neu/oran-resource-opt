import os
import pandas as pd

# Set the base directory path
sub_dirs = ['random', 'round1_tabq', 'round2_deepq_from_tabq']

# List of CSV file names (you can automate this if there are more files)
file_names = [
    '1010123456002_metrics.csv',
    '1010123456003_metrics.csv',
    '1010123456004_metrics.csv',
    '1010123456005_metrics.csv',
    '1010123456006_metrics.csv',
    '1010123456007_metrics.csv',
    '1010123456008_metrics.csv',
    '1010123456009_metrics.csv',
    '1010123456010_metrics.csv'
]

# Directory to save concatenated files
output_dir = 'concatenated'
os.makedirs(output_dir, exist_ok=True)

# Loop through the file names
for file_name in file_names:
    print(f'file name: {file_name}')
    # List to store DataFrames from each sub-directory
    df_list = []
    last_timestamp = 0  # Initialize the last timestamp

    # Loop through each sub-directory in order
    for sub_dir in sub_dirs:
        file_path = os.path.join(sub_dir, file_name)
        # Read the CSV file and append to the list
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, nrows=240)

            # If there is a 'Timestamp' column, adjust it
            if 'Timestamp' in df.columns:
                if last_timestamp > 0:
                    # Get the difference between each timestamp in the current DataFrame
                    timestamp_diff = df['Timestamp'].diff().fillna(0)
                    # Adjust timestamps based on the last timestamp from the previous file
                    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0] + last_timestamp + 250
                # Update last_timestamp to the final timestamp of this DataFrame
                last_timestamp = df['Timestamp'].iloc[-1]

            df_list.append(df)

    # Concatenate all DataFrames for the current file
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Save the concatenated DataFrame to the output directory
    output_file_path = os.path.join(output_dir, file_name)
    concatenated_df.to_csv(output_file_path, index=False)

    print(f'Successfully concatenated and saved: {output_file_path}')
