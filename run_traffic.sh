#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 10 UEs (if you have less than 8, use 'genesys-' as a place holder)


#set -x

out_dir=${1%.*}
experiments=1
eNB_PORT=5305
UE_PORT=6415

echo "using $1, results will be saved in ./$out_dir"
sleep 10
mkdir $out_dir

read -r gnb < $1
echo "gnb is: $gnb"
ric=$(sed '12!d' $1)
echo "ric is $ric"

# Define arrays for user tuples
declare -A user_tuples
user_tuples[1]="1 2 2"
user_tuples[2]="1 1 5"
user_tuples[3]="2 2 2"
user_tuples[4]="1 1 4"
user_tuples[5]="1 5 1"
user_tuples[6]="2 1 3"
user_tuples[7]="1 1 1"
user_tuples[8]="2 4 1"
user_tuples[9]="1 2 3"
user_tuples[10]="3 3 1"
user_tuples[11]="2 3 2"
user_tuples[12]="1 3 3"
user_tuples[13]="2 1 2"
user_tuples[14]="3 1 1"
user_tuples[15]="1 1 3"

# Function to generate random slice configuration
generate_random_slice_config() {
    local tuple=$1
    local slice_mapping=""
    local ue_list=("${@:2}")
    local slice_names=("mmtc" "urllc" "embb")

    # Shuffle UE indices for randomness
    echo "Debug: original UE list: ${ue_list[@]}" >&2
    ue_list=($(echo "${ue_list[@]}" | tr ' ' '\n' | shuf))
    echo "Debug: Shuffled ue_list: ${ue_list[@]}" >&2

    # Split the tuple into an array
    IFS=' ' read -r -a tuple_array <<< "$tuple"
    echo "Debug: Parsed tuple_array: ${tuple_array[@]}" >&2

    # Assign UEs to slices based on the tuple
    local ue_index=0
    for i in "${!tuple_array[@]}"; do
        num_users=${tuple_array[$i]}
        slice=${slice_names[$i]}
        echo "Debug: Assigning $num_users UEs to slice $slice" >&2
        for j in $(seq 1 $num_users); do
            # Stop if we've run out of UEs
            if [[ $ue_index -ge ${#ue_list[@]} ]]; then
                echo "Warning: Not enough UEs to assign all users. Returning partial configuration." >&2
                echo "$slice_mapping"
                return 0
            fi
            slice_mapping="${slice_mapping}${ue_list[$ue_index]}:${slice} "
            ue_index=$((ue_index + 1))
        done
    done

    # Return the complete slice configuration
    echo "$slice_mapping"
}

update_slice_users() {
    local exp=$1
    local config=$2
    echo "Updating slice-users for experiment $exp with config: $config"

    # Split the configuration string into an array
    IFS=' ' read -r -a config_array <<< "$config"

    # Map slice names to slice numbers
    declare -A slice_map
    slice_map[mmtc]=0
    slice_map[urllc]=1
    slice_map[embb]=2

    # Create an array to track which IMSIs need to be updated
    declare -A imsi_updates

    # Loop through the config array and prepare the IMSI and slice mapping
    for i in "${!config_array[@]}"; do
        # Split the config item (e.g., "7:mmtc") into UE ID and slice name
        IFS=':' read -r ue_id slice_name <<< "${config_array[$i]}"

        # Trim leading/trailing whitespace from slice_name
        slice_name=$(echo "$slice_name" | xargs)

        # Check if slice_name is valid
        if [[ -z "$slice_name" ]]; then
            echo "Error: Slice name not found for UE ID $ue_id"
            continue
        fi

        # Validate slice name
        slice_value=${slice_map[$slice_name]}
        if [[ -z "$slice_value" ]]; then
            echo "Error: Invalid slice name $slice_name for UE ID $ue_id"
            continue
        fi

        imsi="001010123456$(printf "%03d" $ue_id)"  # Format IMSI as 0010101234560XX
        imsi_updates["$imsi"]=$slice_value

    done

    # Initialize a variable to hold the entire sed command
    sed_commands=""

    # Now loop through the file and apply the slice updates for the specific IMSIs
    for imsi in "${!imsi_updates[@]}"; do
        slice_value=${imsi_updates[$imsi]}
        # Construct the sed command to update the specific IMSI's slice
        sed_commands="${sed_commands}sed -i '/^$imsi::/s/::[0-9]*/::${slice_value}/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt"$'\n'
    done

    echo "Updating UE slice assignment with $sed_commands"
    # Run the accumulated sed commands in a single SSH session to update the file
    sshpass -p 'scope' ssh "$gnb" "$sed_commands"

    # Sleep to ensure everything is applied
    sleep 5
}


# Updated traffic generation function
traffic_gen() {
    local exp=$1
    local config_file="$2"
    local slice_mapping=$3
    echo "Setting up traffic generation for experiment $exp"

    # Loop through each entry in the slice mapping
    for entry in $slice_mapping; do
        # Parse the entry (e.g., 5:mmtc)
        IFS=':' read -r ue_id slice_type <<< "$entry"

        # Find a random file that matches the slice type
        local random_file=$(ls "./raw" | grep -i "$slice_type" | shuf -n 1)
        local ip=$((ue_id + 1))

        echo "Starting TGEN for $slice_type on UE: $ue_id"
        echo "Using trace: ${random_file}"
        echo "Using IP: 172.16.0.${ip}"

        # Start traffic on the gNB for the given slice type
        sshpass -p 'scope' ssh $gnb "cd TRACTOR && timeout 160 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        sleep 2

        # Start traffic on the UE for the given slice type
        local line=$(sed -n "${ue_id}p" "$config_file")  # Get the UE line from the config file
        sshpass -p 'scope' ssh $line "cd TRACTOR && timeout 158 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &

        # Increment ports and UE counter
        eNB_PORT=$((eNB_PORT + 1))
        UE_PORT=$((UE_PORT + 1))
    done
}

check_connectivity() {
    local ue_node=$1
    local gnb_ip="172.16.0.1"  # Replace with the actual gNB IP if different
    local log_file="connectivity_log.txt"

    echo "Checking connectivity from UE ($ue_node) to gNB ($gnb_ip)..."

    # Try to establish an SSH connection first
    ssh_output=$(sshpass -p 'scope' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $ue_node "echo SSH_SUCCESS" 2>&1)

    if ! echo "$ssh_output" | grep -q "SSH_SUCCESS"; then
        echo "ERROR: Unable to SSH into $ue_node. Connection failed." | tee -a $log_file
        return 1
    fi

    # Execute ping command and capture the output
    ping_output=$(sshpass -p 'scope' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $ue_node "ping -c 4 $gnb_ip 2>&1")

    # Parse the ping results
    if echo "$ping_output" | grep -q "0 received"; then
        echo "ERROR: No packets received. Network connection for $ue_node failed." | tee -a $log_file
        return 1
    elif echo "$ping_output" | grep -q "100% packet loss"; then
        echo "ERROR: 100% packet loss. Network connection for $ue_node failed." | tee -a $log_file
        return 1
    elif echo "$ping_output" | grep -q "Network is unreachable"; then
        echo "ERROR: Network is unreachable. Network connection for $ue_node failed." | tee -a $log_file
        return 1
    else
        # Extract packet loss and latency information
        packet_loss=$(echo "$ping_output" | grep -oP '\d+(?=% packet loss)')
        rtt_stats=$(echo "$ping_output" | grep "rtt" | awk '{print $4}')
        min=$(echo $rtt_stats | cut -d'/' -f1)
        avg=$(echo $rtt_stats | cut -d'/' -f2)
        max=$(echo $rtt_stats | cut -d'/' -f3)

        echo "SUCCESS: Network connection for $ue_node is up." | tee -a $log_file
        echo "Packet loss: $packet_loss%, Min RTT: $min ms, Avg RTT: $avg ms, Max RTT: $max ms"
        return 0
    fi
}


ue_nodes=()

for num in $(seq 2 11); do
  line=$(sed -n "${num}p" "$1")
  echo "Testing network connectivity for $line"
  if check_connectivity $line; then
    ue_nodes+=($num)  # Add to ue_nodes if connectivity succeeds
  fi
  sleep 2
done

if [ ${#ue_nodes[@]} -eq 0 ]; then
    echo "ERROR: No UEs with connectivity. Exiting."
    exit 1
fi

# Print the updated ue_nodes list
echo "Updated ue_nodes list: ${ue_nodes[@]}"

# Main loop to run experiments
while [ $experiments -le 15 ]; do
    echo "Running Experiment $experiments"
    sleep 5

    # Generate slice mapping for the experiment
    tuple=${user_tuples[$experiments]}
    slice_mapping=$(generate_random_slice_config "$tuple" "${ue_nodes[@]}")
    echo "Slice mapping $slice_mapping"

    # Update slice users
    update_slice_users $experiments "$slice_mapping"

    # Set up traffic generation
    traffic_gen $experiments "$1" "$slice_mapping"

    # remove any existing metrics
    echo "Removing any existing metrics"
    sshpass -p "scope" ssh $gnb "
    for file in /root/radio_code/scope_config/metrics/csv/101*_metrics.csv; do
        sed -i '2,\$d' \"\$file\"
    done
    "

    # wait for 2 minutes
    sleep 120
    mkdir -p "$out_dir/$experiments/"
    echo "Updated ue_nodes list: ${ue_nodes[@]}" > "$out_dir/$experiments/slice_mapping_log.txt"
    echo "User tuple - $tuple" >>"$out_dir/$experiments/slice_mapping_log.txt"
    echo "Slice map - $slice_mapping" >> "$out_dir/$experiments/slice_mapping_log.txt"

    # copy log files from the gNB
    sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./"$out_dir/$experiments/"
    # copy log file from xApp
    sshpass -p "ChangeMe" ssh $ric "docker cp sample-xapp-24:/home/xapp-logger.log /root/."
    # Truncate the log file in the Docker container
    # sshpass -p "ChangeMe" ssh $ric "docker exec sample-xapp-24 sh -c 'echo > /home/xapp-logger.log'"
    # Copy the log file to local machine
    sshpass -p "ChangeMe" scp $ric:/root/xapp-logger.log ./"$out_dir/$experiments/"
    # Delete the remote copy of the log file
    sshpass -p "ChangeMe" ssh $ric "rm /root/xapp-logger.log"

    sleep 55
    # Increment the experiment counter
    echo "Finished experiment $experiments"
    experiments=$((experiments + 1))
    clear -x
done


echo "All tests complete"

bash ./process_log_files.sh $out_dir
echo "Cleaned Log Files"
