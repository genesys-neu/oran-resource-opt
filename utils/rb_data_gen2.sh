#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 10 UEs (if you have less than 8, use 'genesys-' as a place holder)


#set -x

out_dir=${1%.*}
experiments=8
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
      echo "Letting the gNB start"
      sleep 20
    fi
    sleep 4
    clear -x
done

#exit 0

sleep 20
clear -x
echo "Started all SRNs"
sleep 20

while [ $experiments -le 10 ]; do
  echo "Running Experiment $experiments"
  # run 10 trials
  trial=1
  while [ $trial -le 10 ]; do
    echo "Running Trial $trial"
    echo "Updating slice-allocation"
    if [ $trial -eq 1 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 2], 1: [3, 8], 2: [9, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 2 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 2], 1: [3, 7], 2: [8, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 3 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 2], 1: [3, 6], 2: [7, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 4 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 1], 1: [2, 8], 2: [9, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 5 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 1], 1: [2, 7], 2: [8, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 6 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 1], 1: [2, 6], 2: [7, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 7 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 1], 1: [2, 5], 2: [6, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 8 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 0], 1: [1, 8], 2: [9, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 9 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 0], 1: [1, 7], 2: [8, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    elif [ $trial -eq 10 ]; then
      sshpass -p "scope" ssh $gnb "sed -i '12s/.*/  \"slice-allocation\": \"{0: [0, 0], 1: [1, 6], 2: [7, 17]}\",/' /root/radio_api/radio_rb_data.conf"
    fi

    if [ $experiments -le 3 ]; then
      # default config is fine, take no action
      echo "Using default slice-users"
      # "slice-users": "{0: [11], 1: [2, 3, 4], 2:[5, 6, 7, 8, 9, 10]}"
      # set up the traffic gen
      for num in $(seq 2 11); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -ge 2 ] && [ $num -le 4 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 5 ] && [ $num -le 10 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 11 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
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

    elif [ $experiments -le 6 ]; then
      # update slice-users
      echo "Updating slice-users"
      sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [11], 1: [2, 3, 4, 5], 2:[6, 7, 8, 9, 10]}\",/' /root/radio_api/radio_rb_data.conf"
      # set up the traffic gen
      for num in $(seq 2 11); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -ge 2 ] && [ $num -le 5 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 6 ] && [ $num -le 10 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 11 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
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

    elif [ $experiments -eq 7 ]; then
      # update slice-users
      echo "Updating slice-users"
      sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [6], 1: [2], 2:[3, 4, 5]}\",/' /root/radio_api/radio_rb_data.conf"
      # set up the traffic gen
      for num in $(seq 2 6); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -eq 2 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 3 ] && [ $num -le 5 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 6 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
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

    elif [ $experiments -eq 8 ]; then
      # update slice-users
      echo "Updating slice-users"
      sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [7], 1: [2], 2:[3, 4, 5, 6]}\",/' /root/radio_api/radio_rb_data.conf"
      # set up the traffic gen
      for num in $(seq 2 7); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -eq 2 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 3 ] && [ $num -le 6 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 7 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
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

    elif [ $experiments -eq 9 ]; then
      # update slice-users
      echo "Updating slice-users"
      sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [7], 1: [2, 3], 2:[4, 5, 6]}\",/' /root/radio_api/radio_rb_data.conf"
      # set up the traffic gen
      for num in $(seq 2 7); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -ge 2 ] && [ $num -le 3 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 4 ] && [ $num -le 6 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 7 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
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

    elif [ $experiments -eq 10 ]; then
      # update slice-users
      echo "Updating slice-users"
      sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [8], 1: [2, 3], 2:[4, 5, 6, 7]}\",/' /root/radio_api/radio_rb_data.conf"
      # set up the traffic gen
      for num in $(seq 2 8); do
        line=$(sed -n "${num}p" "$1")
        echo "Starting TGEN for SRN: $line"
        if [ $num -ge 2 ] && [ $num -le 3 ]; then
          random_file=$(ls "../raw" | grep -i "urll" | shuf -n 1)
          # start URLLC (random trace)
        elif [ $num -ge 4 ] && [ $num -le 7 ]; then
          random_file=$(ls "../raw" | grep -i "embb" | shuf -n 1)
          # start eMBB (random trace)
        elif [ $num -eq 8 ]; then
          random_file=$(ls "../raw" | grep -i "mmtc" | shuf -n 1)
          # start mmtc (random trace)
        fi
        ip=$((num + 1))
        echo "Using trace: ${random_file}"
        echo "Using ip: 172.16.0.${ip}"
        echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
        echo "Starting gNB"
        sshpass -p "scope" ssh $gnb "cd TRACTOR && timeout 150 python traffic_gen.py --eNB -f ./raw/${random_file} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        sleep 2
        echo "Starting UE"
        sshpass -p "scope" ssh $line "cd TRACTOR && timeout 148 python traffic_gen.py -f ./raw/${random_file} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
        eNB_PORT=$((eNB_PORT+1))
        UE_PORT=$((UE_PORT+1))
      done

    fi

    # remove any existing metrics
    sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"
    echo "Removing any existing metrics"
    # wait for 2 minutes
    sleep 120
    mkdir -p "$out_dir/$experiments/$trial"
    #copy log files from the gNB
    sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./"$out_dir/$experiments/$trial/"
    echo "Finished trial $trial, copying files..."
    sleep 20
    # increment the trial counter
    trial=$((trial + 1))
    clear -x
  done
  # Increment the experiment counter
  echo "Finished experiment $experiments"
  experiments=$((experiments + 1))
  clear -x
done

echo "All tests complete"
