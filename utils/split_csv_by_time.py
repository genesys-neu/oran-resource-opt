import os
import pandas as pd

# Path to the folder containing the CSV files
source_folder = '../raw/'

# Desired time interval for each chunk
time_interval = 160  # seconds

# Iterate over all CSV files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        # Load the CSV file
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path)

        # Ensure the 'Time' column is sorted
        df = df.sort_values(by='Time').reset_index(drop=True)

        # Get the maximum time to determine the number of chunks
        max_time = df['Time'].max()

        # Create output folder if it doesn't exist
        output_folder = os.path.join(source_folder, 'processed')
        os.makedirs(output_folder, exist_ok=True)

        # Split into chunks
        chunk_number = 1
        start_time = 0
        while start_time <= max_time:
            # Select data within the current chunk range
            chunk_df = df[(df['Time'] >= start_time) & (df['Time'] < start_time + time_interval)].copy()

            if not chunk_df.empty:
                # Reset 'Time' column: first row becomes 0, keeping the time differences the same
                chunk_df['Time'] = chunk_df['Time'] - chunk_df['Time'].iloc[0]

                # Save the chunk to a new CSV file
                chunk_filename = f"{os.path.splitext(filename)[0]}_part{chunk_number}.csv"
                chunk_path = os.path.join(output_folder, chunk_filename)
                chunk_df.to_csv(chunk_path, index=False)

                print(f"Created: {chunk_filename}")

                # Increment chunk number
                chunk_number += 1

            # Move to the next 160-second window
            start_time += time_interval
