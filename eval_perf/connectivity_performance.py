import re
from collections import defaultdict

# Filepath for the log file
log_file = "../connectivity_log.txt"
output_file = "network_statistics.txt"

# Dictionaries to count successes and failures per node
success_count = defaultdict(int)
failure_count = defaultdict(int)

# Parse the log file
with open(log_file, "r") as file:
    for line in file:
        # Match lines for success
        success_match = re.match(r"SUCCESS: Network connection for (genesys-\d+) is up\.", line)
        if success_match:
            node = success_match.group(1)
            success_count[node] += 1
        # Match lines for failure
        failure_match = re.match(r"ERROR: .* Network connection for (genesys-\d+) failed\.", line)
        if failure_match:
            node = failure_match.group(1)
            failure_count[node] += 1

# Write statistics to the output file
with open(output_file, "w") as output:
    output.write("Node Statistics:\n")
    for node in sorted(set(success_count.keys()).union(failure_count.keys())):
        total_attempts = success_count[node] + failure_count[node]
        if total_attempts > 0:
            success_percent = (success_count[node] / total_attempts) * 100
        else:
            success_percent = 0  # Avoid division by zero
        output.write(f"{node}: {success_percent:.0f}% success with {total_attempts} attempts\n")

print(f"Statistics saved to {output_file}")
