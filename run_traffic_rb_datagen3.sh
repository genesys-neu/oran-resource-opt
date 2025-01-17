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

# Define arrays for user tuples and slice-user configurations
declare -A user_tuples
user_tuples[1]="3 2 3"
user_tuples[2]="2 3 4"
user_tuples[3]="1 2 5"
user_tuples[4]="1 2 1"
user_tuples[5]="1 1 4"
user_tuples[6]="0 2 2"
user_tuples[7]="0 1 2"
user_tuples[8]="1 2 3"
user_tuples[9]="1 1 2"
user_tuples[10]="3 3 3"
user_tuples[11]="1 3 5"
user_tuples[12]="2 3 5"
user_tuples[13]="3 2 4"
user_tuples[14]="3 3 4"
user_tuples[15]="2 4 4"
# Add more user tuples as needed for other experiments

declare -A slice_config
slice_config[1]="2::0 3::0 4::0 5::1 6::1 7::2 8::2 9::2"
slice_config[2]="2::0 3::0 4::1 5::1 6::1 7::2 8::2 9::2 10::2"
slice_config[3]="2::0 3::1 4::1 5::2 6::2 7::2 8::2 9::2"
slice_config[4]="2::0 3::1 4::1 5::2"
slice_config[5]="2::0 3::1 4::2 5::2 6::2 7::2"
slice_config[6]="2::1 3::1 4::2 5::2"
slice_config[7]="2::1 3::2 4::2"
slice_config[8]="2::0 3::1 4::1 5::2 6::2 7::2"
slice_config[9]="2::0 3::1 4::2 5::2"
slice_config[10]="2::0 3::0 4::0 5::1 6::1 7::1 8::2 9::2 10::2"
slice_config[11]="2::0 3::1 4::1 5::1 6::2 7::2 8::2 9::2 10::2"
slice_config[12]="2::0 3::0 4::1 5::1 6::1 7::2 8::2 9::2 10::2 11::2"
slice_config[13]="2::0 3::0 4::0 5::1 6::1 7::2 8::2 9::2 10::2"
slice_config[14]="2::0 3::0 4::0 5::1 6::1 7::1 8::2 9::2 10::2 11::2"
slice_config[15]="2::0 3::0 4::1 5::1 6::1 7::1 8::2 9::2 10::2 11::2"
# Add more configurations for other experiments


# Function to update slice-users
update_slice_users() {
    local exp=$1
    local config=${slice_config[$exp]}
    # echo "config: $config"
    echo "Updating slice-users for experiment $exp"

    # Split the configuration string into an array
    IFS=' ' read -r -a config_array <<< "$config"

    # Initialize a variable to hold the entire sed command
    sed_commands=""

    # Loop through the config array and build the sed commands
    for i in "${config_array[@]}"; do
        index=$(echo "$i" | cut -d: -f1)
        value=$(echo "$i" | cut -d: -f3)

        # Add each sed command to the sed_commands string, each on a new line
        sed_commands="${sed_commands}sed -i '${index}s/::.*/::${value}/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt"$'\n'
    done

    # Run the accumulated sed commands in a single SSH session
    # echo "sshpass -p 'scope' ssh $gnb \"$sed_commands\""
    sshpass -p 'scope' ssh $gnb "$sed_commands"

    # Sleep to ensure everything is applied
    sleep 5
}

# Function to set up traffic generation based on user tuples
traffic_gen() {
    local exp=$1
    echo "Setting up traffic generation for experiment $exp"
    local config_file="$2"
    # Get the user tuple for this experiment (e.g., "1 3 4")
    local tuple=${user_tuples[$exp]}
    # echo "local tuple $tuple"

    # Split the tuple into an array
    local tuple_array=($tuple)

    # First number in the tuple corresponds to MMTC
    mmtc_users=${tuple_array[0]}
    echo "mmtc users: $mmtc_users"
    # Second number in the tuple corresponds to URLLC
    urllc_users=${tuple_array[1]}
    echo "urllc users: $urllc_users"
    # Third number in the tuple corresponds to eMBB
    embb_users=${tuple_array[2]}
    echo "embb users: $embb_users"

    ue_counter=2 # UE lines start from line 2

    # Assign MMTC traffic
    for i in $(seq 1 $mmtc_users); do
        line=$(sed -n "${ue_counter}p" "$config_file")
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
        ip=$((ue_counter + 1))
        echo "Starting TGEN for MMTC on SRN: $line"
        echo "Using trace: ${random_file}"
        echo "Using ip: 172.16.0.${ip}"
        sshpass -p 'scope' ssh $gnb "cd TRACTOR && timeout 160 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        sleep 2
        sshpass -p 'scope' ssh $line "cd TRACTOR && timeout 158 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        eNB_PORT=$((eNB_PORT+1))
        UE_PORT=$((UE_PORT+1))
        ue_counter=$((ue_counter+1))
    done

    # Assign URLLC traffic
    for i in $(seq 1 $urllc_users); do
        line=$(sed -n "${ue_counter}p" "$config_file")
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        ip=$((ue_counter + 1))
        echo "Starting TGEN for URLLC on SRN: $line"
        echo "Using trace: ${random_file}"
        echo "Using ip: 172.16.0.${ip}"
        sshpass -p 'scope' ssh $gnb "cd TRACTOR && timeout 160 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        sleep 2
        sshpass -p 'scope' ssh $line "cd TRACTOR && timeout 158 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        eNB_PORT=$((eNB_PORT+1))
        UE_PORT=$((UE_PORT+1))
        ue_counter=$((ue_counter+1))
    done

    # Assign eMBB traffic
    for i in $(seq 1 $embb_users); do
        line=$(sed -n "${ue_counter}p" "$config_file")
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        ip=$((ue_counter + 1))
        echo "Starting TGEN for eMBB on SRN: $line"
        echo "Using trace: ${random_file}"
        echo "Using ip: 172.16.0.${ip}"
        sshpass -p 'scope' ssh $gnb "cd TRACTOR && timeout 160 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        sleep 2
        sshpass -p 'scope' ssh $line "cd TRACTOR && timeout 158 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        eNB_PORT=$((eNB_PORT+1))
        UE_PORT=$((UE_PORT+1))
        ue_counter=$((ue_counter+1))
    done
}


while [ $experiments -le 15 ]; do
  echo "Running Experiment $experiments"
  sleep 5
  update_slice_users $experiments
  traffic_gen $experiments "$1"

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


  sleep 45
  # Increment the experiment counter
  echo "Finished experiment $experiments"
  experiments=$((experiments + 1))
  clear -x
done

echo "All tests complete"

bash ./process_log_files.sh $out_dir
echo "Cleaned Log Files"
