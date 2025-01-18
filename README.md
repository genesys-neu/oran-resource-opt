# Dynamic Open RAN Optimization Framework in Colosseum 

## Introduction
This repository focuses on a rapidly deployable Open RAN framework with automated setup and experiment execution in the Colosseum testbed.
This repository automates the setup for running traffic and interference simulations in a 5G network environment using Bash scripts. It streamlines the configuration of various network components and initiates the necessary actions to simulate network traffic and interference.

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


## xAPPs
This repository includes different example xApps. Each of the xApps supports the collection of metrics from multiple UEs, and different control actions.

**TODO: Add a list of possible KPIs that are used**

**TODO: Add the control actions that are supported**

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

### Key Points:
- Deletes small CSV files (≤ 10 rows).
- Trims log files to keep only new entries.


## Disclaimer
These scripts are provided as-is and may require customization based on individual network configurations and environment setups.

