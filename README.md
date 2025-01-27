# Dynamic Open RAN Optimization Framework using Colosseum 

## Introduction
This repository focuses on a rapidly deployable Open RAN framework with automated setup and experiment execution in the Colosseum testbed.
This repository automates the setup for running traffic and interference simulations in a cellular ORAN network environment using Bash scripts. It streamlines the configuration of various network components and initiates the necessary actions to simulate network traffic and interference.

> **Note**: When cloning this repository, ensure you include the submodules by running the following command:
> ```bash
> git clone --recurse-submodules <repository-url>
> ```
> This is critical as some scripts rely on transferring files from your local machine to a remote server (using `scp` or `rsync`). Without the submodules, these scripts will fail.


You must first make a reservation in Colosseum for 2 to 12 nodes. Use the following nodes:
- IMPACT-gNB for the gNB
- IMPACT-UE for 1-10 UEs
- IMPACT-RIC for the near-RT RIC

## Usage
To run an experiment, execute the following command in the terminal:
```
/run_experiment.sh config_file.txt
```
This script automates all the setup and configuration tasks, reducing the manual effort required to prepare the network environment.


## Configuration File
The configuration file (config_file.txt) must list all the nodes for your experiment. It should contain 12 lines. Any unused nodes will be 'genesys-'.
- `config_file.txt`: This file contains the configuration details for the network setup, including information about the gNB, UEs, interferer, observer, and near-RT RIC.
- Ensure that the gNB is listed as the first SRN.
- The subsequent lines in the configuration file should list the UEs (8 lines, but you  may leave the SRN number blank if you are using less)
- The last line must be the near-RT RIC
  

## Execution Steps
The `run_experiment.sh` script will automatically perform the following actions:

1. **Channel Setup**: Initiates the communication channel using the Colosseum CLI to start the RF interface for the experiment.
2. **SRN Startup**: Starts the gNB and UEs as specified in the configuration file. It ensures that the correct files and scripts are transferred and executed on each SRN.
3. **Near-RT RIC Setup**: Configures the near-RT RIC and establishes the E2 interface between the gNB and RIC.
4. **xApp Deployment**: Deploys the xApp (a sample application for resource management) on the near-RT RIC, allowing dynamic traffic classification and network resource allocation.
5. **Traffic Simulation**: Executes the traffic simulation script (`run_traffic.sh`), which generates network traffic based on the configuration, including user distribution and traffic patterns.
6. **Process Log Files**: Automatically collects logs and performs cleanup tasks to ensure a clean environment for subsequent experiments.


## xApp Framework
This repository includes different example xApps. Each of the xApps supports the collection of metrics from multiple UEs, and different control actions. The xApp framework is designed to support many different types of ML models.

### `ue_data` Dictionary
The `ue_data` dictionary contains the following keys:

- **`last_timestamp`**: Tracks the last recorded timestamp for the UE.
- **`kpi_slice`**: Indicates the slice to which the UE is assigned.
- **`kpi_prb`**: Stores the Physical Resource Blocks (PRBs) allocated to the UE.
- **`stale_counter`**: Tracks the number of stale data occurrences for the UE.
- **`inference_kpi`**: A list storing KPIs for use in machine learning inference.

### Available KPIs

- **Downlink KPIs**
  - `dl_mcs`: Downlink Modulation and Coding Scheme (MCS).
  - `dl_n_samples`: Number of downlink samples.
  - `dl_buffer [bytes]`: Size of the downlink buffer in bytes.
  - `tx_brate downlink [Mbps]`: Downlink transmission bitrate in Mbps.
  - `tx_pkts downlink`: Number of downlink transmitted packets.
  - `tx_errors downlink (%)`: Percentage of errors in downlink transmissions.
  - `dl_cqi`: Downlink Channel Quality Indicator (CQI).

- **Uplink KPIs**
  - `ul_mcs`: Uplink Modulation and Coding Scheme (MCS).
  - `ul_n_samples`: Number of uplink samples.
  - `ul_buffer [bytes]`: Size of the uplink buffer in bytes.
  - `rx_brate uplink [Mbps]`: Uplink reception bitrate in Mbps.
  - `rx_pkts uplink`: Number of uplink received packets.
  - `rx_errors uplink (%)`: Percentage of errors in uplink receptions.
  - `ul_sinr`: Uplink Signal-to-Interference-plus-Noise Ratio (SINR).

- **Additional KPIs**
  - `phr`: Power Headroom Report.
  - `sum_requested_prbs`: Total number of requested Physical Resource Blocks (PRBs).
  - `sum_granted_prbs`: Total number of granted Physical Resource Blocks (PRBs).
  - `ul_turbo_iters`: Number of turbo decoding iterations for uplink transmissions.

### Key Functions and Features

1. **`count_ue_assignments()`**
   - Counts the number of UEs assigned to each slice (mMTC, URLLC, and eMBB).
   - Returns:
     - A dictionary of slice counts.
     - A dictionary of PRBs allocated to each slice.

2. **`initialize_agent()` / `initialize_model()`**
   - Creates and returns an agent or model instance loaded from a model definition file.
   - Supports various agent/model types with pre-trained models loaded from specified paths.

3. **External Imports**
   - Models, agents, and utility functions are imported from other scripts and libraries.

4. **Logging**
   - The logging framework is extensively used for:
     - Tracking function execution.
     - Key information and debugging.
     - Recording warnings or errors.
   - Logs are automatically copied to the local machine during execution.

### Control Messages

The xApps support sending control messages to manage scheduling, PRB allocation, UE slice assignments, MCS settings, and power adjustments. The control message has the following structure:

#### Control Message Format

`<scheduling>\n<prb_assignment>\n<ue_slice_assignment>\n<mcs_adjustment>\n<power_adjustment>END`

#### Breakdown of Each Line:

1. **Scheduling (First Line)**  
   Specifies the scheduling algorithm to use:
   - `0`: Round-robin.
   - `1`: Water filling.
   - `2`: Proportionally fair.

2. **PRB Assignment to Slices (Second Line)**  
   Defines the PRB allocation for each slice as a bit mask.  
   Format: `<slice_0_bits>,<slice_1_bits>,<slice_2_bits>`  
   - The sum of PRBs bits across slices must not exceed 17.

3. **UE Slice Assignment (Third Line)**  
   Maps UEs to slices using their IMSI (International Mobile Subscriber Identity).  
   Format: `<imsi>::<slice ID>`  
   - Slice IDs in the example xApps:
     - `0`: mMTC.
     - `1`: URLLC.
     - `2`: eMBB.

4. **MCS Adjustment (Fourth Line)**  
   Specifies the Modulation and Coding Scheme (MCS) to use:  
   Format: `<imsi>::<MCS>`  
   - `0`: Default adaptive modulation.
   - `1`: QPSK.
   - `2`: 16-QAM.
   - `3`: 64-QAM.

5. **Power (Gain) Adjustment (Fifth Line)**  
   Adjusts transmission power:  
   Format: `<imsi>::<gain>`  

#### Example Control Message:
`0,1,2\n3,5,9\n<imsi_1>::0\n<imsi_1>::1\n<imsi_1>::10END`

Note that lines may be skipped if not needed.


### Example xApps
1. `run_xapp_rb_only.py` - This xApp tracks how many users are in each slice and updates the PRB allocation among slices.
2. `run_xapp_tractor_rb.py` - In addition to PRB optimization based on the number of users in a slice, this xApp also performs traffic classification for each UE and will move the UE to the proper slice based on its current traffic pattern.


## Traffic Generation
The `run_experiment.sh` script calls a traffic generation script in the final step. There are two example traffic generation scripts. 
1. `run_traffic.sh` - In this script you can declare the number of users you want in each slice for each trial as a user tuple like `user_tuples[1]="2 3 4"`. The script will randomly assign UEs to the correct slice, and then randomly generate the correct traffic type for each UE.
2. `run_traffic_deterministic.sh` - The main difference for this traffic generator is that it deterministically selects the traffic traces instead of randomly.

It also automates the collection of logs from each trial in the experiment. 


### Output File Structure and Log Collection

Each experiment is saved in a directory named after the `config_file`. Inside this directory, each trial is stored in a sub-directory named `/x`, where `x` is the trial number.

- **Log Files**:
  - **UE Metrics**: Each trial contains individual UE metrics, saved as `101*_metrics.csv` files under the respective trial sub-directory.
  - **xApp Logs**: The xApp logs from the RIC are saved in the trial sub-directory as `xapp-logger.log`.

- **Metadata**: For each trial, a `slice_mapping_log.txt` file is generated, containing:
  - The user tuple used for the trial (e.g., `user tuple - 2 3 4`).
  - The slice mapping configuration for that trial (e.g., `Slice map - 1:mmtc 2:embb 3:urllc`).


## Process Log Files
The `process_log_files.sh` script processes log files and removes small CSV files from each trial directory. It performs the following tasks:

1. **Directory Iteration**: Iterates through each sub-directory (trial) in the main directory and processes the log and CSV files.
2. **CSV File Handling**: Deletes `.csv` files with 10 or fewer rows to avoid unnecessary files.
3. **Log File Processing**: 
   - Trims the `xapp-logger.log` file by deleting the first `PREV_LINE_COUNT` lines if specified, ensuring only new logs are kept.
   - Reports the current number of lines in the log file and checks that the previous line count doesn’t exceed the current line count.
4. **Final Report**: Outputs the final line count from the last sub-directory's log file.


## Traffic Classification
All Traffic Classification training and analysis files can be found in the python subdirectory.


## Utils
The utils subdirectory contains several utility scripts primarily focused on the PRB optimization. All RL policies related to PRB optimization are found here.


## Traffic Traces
All the traffic traces used for both traffic classification and PRB optimization are in the raw subdirectory.


## Disclaimer
These scripts are provided as-is and may require customization based on individual network configurations and environment setups.

