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
# Start the channel
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'

# start the gNB and UEs
for num in $(seq 1 11); do
    line=$(sed -n "${num}p" "$1")
    echo "Starting SRN: $line"
    sshpass -p "scope" scp radio_rb_data.conf $line:/root/radio_api/
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_rb_data.conf" &
    if [ $line = $gnb ]
    then
      sshpass -p "scope" scp adjust_rb.py $gnb:/root/adjust_rb.py
      echo "Letting the gNB start"
      sleep 30
    fi
    sleep 5
    clear -x
done

#exit 0

sleep 30
clear -x
echo "Started all SRNs"
sleep 30

while [ $experiments -le 10 ]; do
  echo "Running Experiment $experiments"

  # Define the number of slices and total users
  total_users=10
  num_slices=3

  # Randomly generate user assignments
  users=($(seq 2 11))  # User numbers from 2 to 11
  random_users=($(shuf -e "${users[@]}"))

  # Generate slice assignments
  slice_assignments=()
  for (( i=0; i<num_slices; i++ )); do
      slice_assignments[$i]=()
  done

  # Distribute users to slices
  for (( i=0; i<total_users; i++ )); do
      slice=$((i % num_slices))
      slice_assignments[$slice]+="${random_users[$i]} "
  done
  echo "Slice Assignments: ${slice_assignments[@]}"

  # Generate the sed commands for SSH
  commands=""
  for (( i=0; i<num_slices; i++ )); do
      for user in ${slice_assignments[$i]}; do
          line_number=$((user))
          commands+="sed -i '${line_number}s/::.*/::${i}/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \\"
      done
  done

  # Remove the trailing '&& \'
  commands=$(echo "$commands" | sed 's/ && \\$//')
  echo "Generated sed commands: $commands"

  # Execute the SSH command
  sshpass -p "scope" ssh $gnb "
  $commands
  "

  # Generate and print the tuple of user counts per slice
  user_counts=()
  for (( i=0; i<num_slices; i++ )); do
      count=$(echo "${slice_assignments[$i]}" | wc -w)
      user_counts+=($count)
  done

  user_tuple=$(IFS=,; echo "${user_counts[*]}")
  echo "Starting RB adjustment for: $user_tuple"

  # Define the trace files for each slice
  slice_trace=("mmtc" "urll" "embb")

  # Set up the traffic gen
  for num in $(seq 2 11); do
      # Determine the slice for the current user
      slice_number=$(for i in ${!slice_assignments[@]}; do
          if echo "${slice_assignments[$i]}" | grep -q "\b${num}\b"; then
              echo $i
          fi
      done)

      # Select the trace file based on slice
      trace_type=${slice_trace[$slice_number]}
      random_file=$(ls "../raw" | grep -i "$trace_type" | shuf -n 1)

      # Print selected trace file
      echo "Starting TGEN for user $num (slice $slice_number) with trace: ${random_file}"
      ip=$((num + 1))
      echo "Using trace: ${random_file}"
      echo "Using ip: 172.16.0.${ip}"
      echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
      echo "Starting gNB"
      sshpass -p "scope" ssh $gnb "cd TRACTOR && timeout 160 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
      sleep 2
      echo "Starting UE"
      sshpass -p "scope" ssh $line "cd TRACTOR && timeout 158 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
      eNB_PORT=$((eNB_PORT+1))
      UE_PORT=$((UE_PORT+1))
  done

  sshpass -p "scope" ssh $gnb "timeout 130 python /root/adjust_rb.py --users $user_tuple" &
  # remove any existing metrics
  # sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"
  echo "Removing any existing metrics"
  sshpass -p "scope" ssh $gnb "
  for file in /root/radio_code/scope_config/metrics/csv/101*_metrics.csv; do
      sed -i '2,\$d' \"\$file\"
  done
  "

  # wait for 2 minutes
  sleep 120
  mkdir -p "$out_dir/$experiments/"
  #copy log files from the gNB
  sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./"$out_dir/$experiments/"

  sleep 30
  # Increment the experiment counter
  echo "Finished experiment $experiments"
  experiments=$((experiments + 1))
  clear -x
done

echo "All tests complete"
