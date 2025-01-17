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

while [ $experiments -le 10 ]; do
  echo "Running Experiment $experiments"

  if [ $experiments -eq 1 ]; then
    # default config is fine, take no action
    echo "Using default slice-users"
    sleep 10
    # "slice-users": "{0: [11], 1: [2, 3, 4], 2:[5, 6, 7, 8, 9, 10]}"
    user_tuple="1 3 6"
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ $num -ge 2 ] && [ $num -le 4 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 5 ] && [ $num -le 10 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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

  elif [ $experiments -eq 2 ]; then
  # default config is fine, take no action
  echo "Using default slice-users"
  sleep 10
  # "slice-users": "{0: [11], 1: [2, 3, 4], 2:[5, 6, 7, 8, 9, 10]}"
  user_tuple="1 3 6"
  # set up the traffic gen
  for num in $(seq 2 11); do
    line=$(sed -n "${num}p" "$1")
    echo "Starting TGEN for SRN: $line"
    if [ $num -ge 2 ] && [ $num -le 4 ]; then
      random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
      # start URLLC (random trace)
    elif [ $num -ge 5 ] && [ $num -le 10 ]; then
      random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
      # start eMBB (random trace)
    elif [ $num -eq 11 ]; then
      random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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

  elif [ $experiments -eq 3 ]; then
    # update slice-users
    echo "Updating slice-users"
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [11], 1: [2, 3], 2:[4, 5, 6, 7, 8, 9, 10]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="1 2 7"
    sshpass -p "scope" ssh $gnb "
      sed -i '8s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '6s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ $num -ge 2 ] && [ $num -le 3 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 4 ] && [ $num -le 10 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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

  elif [ $experiments -eq 4 ]; then
    # update slice-users
    echo "Updating slice-users"
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [11], 1: [2, 3, 4, 5], 2:[6, 7, 8, 9, 10]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="1 4 5"
    sshpass -p "scope" ssh $gnb "
      sed -i '11s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '6s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ $num -ge 2 ] && [ $num -le 5 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 6 ] && [ $num -le 10 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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

  elif [ $experiments -eq 5 ]; then
    # update slice-users
    echo "Updating slice-users"
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [6, 7], 1: [2, 8, 9], 2:[3, 4, 5, 10, 11]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="2 3 5"
    sshpass -p "scope" ssh $gnb "
      sed -i '6s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ "$num" -eq 2 ] || [ "$num" -eq 8 ] || [ "$num" -eq 9 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif { [ $num -ge 3 ] && [ $num -le 5 ]; } || [ $num -eq 10 ] || [ "$num" -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 6 ] && [ $num -le 7 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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

  elif [ $experiments -eq 6 ]; then
    # update slice-users
    echo "Updating slice-users"
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [6, 7], 1: [2, 8, 9, 10], 2:[3, 4, 5, 11]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="2 4 4"
    sshpass -p "scope" ssh $gnb "
      sed -i '6s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ "$num" -eq 2 ] || [ "$num" -eq 8 ] || [ "$num" -eq 9 ] || [ "$num" -eq 10 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif { [ $num -ge 3 ] && [ $num -le 5 ]; } || [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 6 ] && [ $num -le 7 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [6, 7, 8], 1: [2, 9, 10], 2:[3, 4, 5, 11]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="3 3 4"
    sshpass -p "scope" ssh $gnb "
      sed -i '6s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ "$num" -eq 2 ] || [ "$num" -eq 9 ] || [ "$num" -eq 10 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif { [ $num -ge 3 ] && [ $num -le 5 ]; } || [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 6 ] && [ $num -le 8 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [7, 8, 9], 1: [2, 10, 11], 2:[3, 4, 5, 6]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="3 3 4"
    sshpass -p "scope" ssh $gnb "
      sed -i '7s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '6s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ "$num" -eq 2 ] || [ "$num" -eq 10 ] || [ "$num" -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 3 ] && [ $num -le 6 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 7 ] && [ $num -le 9 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [7, 8, 9], 1: [2, 3, 10, 11], 2:[4, 5, 6]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="3 4 3"
    sshpass -p "scope" ssh $gnb "
      sed -i '7s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '8s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '6s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ $num -eq 2 ] || [ $num -eq 3 ] || [ $num -eq 10 ] || [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 4 ] && [ $num -le 6 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 7 ] && [ $num -le 9 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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
    # sshpass -p "scope" ssh $gnb "sed -i '13s/.*/  \"slice-users\": \"{0: [8, 9, 10], 1: [2, 3], 2:[4, 5, 6, 7, 11]}\",/' /root/radio_api/radio_rb_data.conf"
    user_tuple="3 2 5"
    sshpass -p "scope" ssh $gnb "
      sed -i '8s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '9s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '10s/::.*/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '2s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '3s/::.*/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '4s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '5s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '6s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '7s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt && \
      sed -i '11s/::.*/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt
    "
    sleep 10
    # set up the traffic gen
    for num in $(seq 2 11); do
      line=$(sed -n "${num}p" "$1")
      echo "Starting TGEN for SRN: $line"
      if [ $num -ge 2 ] && [ $num -le 3 ]; then
        random_file=$(ls "./raw" | grep -i "urll" | shuf -n 1)
        # start URLLC (random trace)
      elif [ $num -ge 4 ] && [ $num -le 7 ] || [ $num -eq 11 ]; then
        random_file=$(ls "./raw" | grep -i "embb" | shuf -n 1)
        # start eMBB (random trace)
      elif [ $num -ge 8 ] && [ $num -le 10 ]; then
        random_file=$(ls "./raw" | grep -i "mmtc" | shuf -n 1)
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
  fi

  echo "Starting RB adjustment with users $user_tuple"
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


  sleep 30
  # Increment the experiment counter
  echo "Finished experiment $experiments"
  experiments=$((experiments + 1))
  clear -x
done

echo "All tests complete"
