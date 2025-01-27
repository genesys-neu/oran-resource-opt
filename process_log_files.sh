#!/bin/bash


# Initialize the directory path (assuming current directory)
MAIN_DIR="$1"

# Initialize variables
# Check if an argument is provided for the initial previous line count
if [ -n "$2" ]; then
  PREV_LINE_COUNT="$2"
else
  PREV_LINE_COUNT=0
fi

# Ensure the provided argument is a valid number
if ! [[ "$PREV_LINE_COUNT" =~ ^[0-9]+$ ]]; then
  echo "Error: The initial line count must be a non-negative integer."
  exit 1
fi
LOG_FILE_NAME="xapp-logger.log"
FINAL_LINE_COUNT=0

# Loop through subdirectories in numerical order
for DIR in $(ls -d $MAIN_DIR/[0-9]* | sort -V); do
  # Construct the path to the log file in the current subdirectory
  LOG_FILE="$DIR/$LOG_FILE_NAME"

  # Check and process .csv files in the current directory
  for CSV_FILE in "$DIR"/*.csv; do
    if [ -f "$CSV_FILE" ]; then
      # Get the number of rows in the .csv file
      ROW_COUNT=$(wc -l < "$CSV_FILE")
      if [ "$ROW_COUNT" -le 10 ]; then
        echo "Deleting $CSV_FILE: Only $ROW_COUNT rows."
        rm "$CSV_FILE"
      fi
    fi
  done

  # Check if the log file exists in the current directory
  if [ -f "$LOG_FILE" ]; then
    # Get the current number of lines in the log file
    CURRENT_LINE_COUNT=$(wc -l < "$LOG_FILE")
    echo "Processing $LOG_FILE: $CURRENT_LINE_COUNT lines"

    # If this is not the first directory, delete the previous lines
    if [ "$PREV_LINE_COUNT" -gt 0 ]; then
      if [ "$PREV_LINE_COUNT" -lt "$CURRENT_LINE_COUNT" ]; then
        # Delete lines from 1 to PREV_LINE_COUNT
        sed -i "1,${PREV_LINE_COUNT}d" "$LOG_FILE"
        echo "Deleted first $PREV_LINE_COUNT lines from $LOG_FILE"
      else
        echo "Error: Previous line count ($PREV_LINE_COUNT) is greater than or equal to current file lines ($CURRENT_LINE_COUNT). Skipping deletion."
      fi
    fi

    # Update the previous line count
    PREV_LINE_COUNT=$CURRENT_LINE_COUNT

  else
    echo "Log file $LOG_FILE not found, skipping."
  fi
done

# Report the final line count from the last subdirectory
echo "Final line count of the last subdirectory: $CURRENT_LINE_COUNT"
