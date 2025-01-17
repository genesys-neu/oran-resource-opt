# Dynamic Open RAN Optimization Framework in Colosseum 

## Introduction
This repository focuses on a rapidly deployable Open RAN framework with automated setup and experiment execution in the Colosseum testbed.
This Bash script automates the setup process for running traffic and interference simulations in a 5G network environment. It streamlines the configuration of various network components and initiates the necessary actions to simulate network traffic and interference.

You must first make a reservation in Colosseum for 2 to 12 nodes. Use the following nodes:
- IMPACT-gNB for the gNB
- IMPACT-UE for 1-8 UEs
- IMPACT-RIC for the near-RT RIC

## Usage
To run the script, execute the following command in the terminal:
```
/run_IMPACT_new.sh config_file.txt [trace_name.csv]
```
- `config_file.txt`: This file contains the configuration details for the network setup, including information about the gNB, UEs, interferer, observer, and near-RT RIC.
- `[trace_name.csv]` (optional): Specifies an optional trace file to be used for the simulation.

## Configuration File
You must list all the nodes for your experiment in the configuration file (`config_file.txt`). The config file should have 12 lines. Any unused nodes will be 'genesys-'.
- Ensure that the gNB is listed as the first SRN.
- The subsequent lines in the configuration file should list the UEs (8 lines, but you  may leave the SRN number blank if you are using less)
- The interferer
- The observer
- The near-RT RIC
  
## Interference Setup
- The script sets up interference by transmitting tones using Universal Hardware Driver (UHD) tools.
- The interference source (interferer) is specified in the configuration file.

## Execution Steps
1. **Channel Setup**: Initiates the channel for communication using Colosseum CLI.
2. **Interference Configuration**: Configures interference by transmitting tones from the interferer.
3. **SRN Startup**: Starts the gNB and UEs specified in the configuration file.
4. **Near-RT RIC Setup**: Sets up the near-RT RIC and establishes the E2 interface.
5. **xApp Deployment**: Deploys the xApp (sample application) on the near-RT RIC.
6. **Listener Activation**: Starts the listener to monitor network activity.
7. **Traffic Simulation**: Executes the traffic simulation script.
8. **Completion**: Indicates the completion of all tests.

## Additional Notes
- The script automates various setup and configuration tasks, reducing the manual effort required to prepare the network environment.
- Ensure that the necessary dependencies, such as `sshpass` and `UHD tools`, are installed on the system before running the script.
- Customize the script parameters and configurations as needed to suit specific network simulation requirements.

## Disclaimer
This script is provided as-is and may require customization based on individual network configurations and environment setups. Use it at your own risk, and always review and test the script in a safe environment before deploying it in a production or critical network scenario.

