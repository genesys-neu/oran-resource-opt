import pandas as pd

# Define your column headers
COLUMN_HEADERS = [
    'dl_mcs', 'dl_n_samples', 'dl_buffer [bytes]', 'tx_brate downlink [Mbps]',
    'tx_pkts downlink', 'tx_errors downlink (%)', 'dl_cqi', 'ul_mcs',
    'ul_n_samples', 'ul_buffer [bytes]', 'rx_brate uplink [Mbps]',
    'rx_pkts uplink', 'rx_errors uplink (%)', 'ul_sinr', 'phr',
    'sum_requested_prbs', 'sum_granted_prbs', 'ul_turbo_iters', 'Label'
]

# Path to the large CSV file without headers
input_file = 'combined_metrics.csv'
output_file = 'combined_metrics.csv'

# Read the CSV file without headers
df = pd.read_csv(input_file, header=None)

# Assign the correct headers to the DataFrame
df.columns = COLUMN_HEADERS

# Save the DataFrame with headers back to a new CSV file
df.to_csv(output_file, index=False)

print(f"Headers added back and saved to {output_file}")
