#!/bin/bash

# Base directory containing the subdirectories
base_directory="../test_config"

# Loop through each subdirectory numbered 1 through 10
for subdir in "$base_directory"/{1..10}; do
    # Check if the subdirectory exists
    if [ -d "$subdir" ]; then
        echo "Processing files in $subdir"

        # Loop through each CSV file in the current subdirectory
        for file in "$subdir"/*.csv; do
            # Check if the file exists
            if [ -f "$file" ]; then
                # Create a temporary file to store the processed data
                temp_file=$(mktemp)

                # Extract the first row and the last 480 rows and save them to the temporary file
                {
                    head -n 1 "$file"          # Get the first row
                    tail -n 480 "$file"        # Get the last 480 rows
                } > "$temp_file"

                # Overwrite the original file with the processed data
                mv "$temp_file" "$file"

                echo "Processed $file"
            else
                echo "No CSV files found in $subdir"
            fi
        done
    else
        echo "$subdir is not a directory"
    fi
done
