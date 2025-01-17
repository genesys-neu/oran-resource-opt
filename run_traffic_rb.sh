#!/bin/bash
## HOW TO RUN: ./run_traffic_rb.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 10 UEs (if you have less than 10, use 'genesys-' as a place holder)
## The RIC will be the 12th line


out_dir=${1%.*}
experiments=1
eNB_PORT=5305
UE_PORT=6415

read -r gnb < $1
echo "gnb is: $gnb"

ric=$(sed '12!d' $1)


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
    random_file=$(ls "raw" | grep -i "urll" | shuf -n 1)
    # start URLLC (random trace)
  elif [ $num -ge 4 ] && [ $num -le 10 ]; then
    random_file=$(ls "raw" | grep -i "embb" | shuf -n 1)
    # start eMBB (random trace)
  elif [ $num -eq 11 ]; then
    random_file=$(ls "raw" | grep -i "mmtc" | shuf -n 1)
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

sleep 120

echo "All traffic complete"


#copy log files from the gNB
sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/*_metrics.csv ./$out_dir/

#copy log file from xApp
sshpass -p "ChangeMe" ssh $ric "docker cp sample-xapp-24:/home/xapp-logger.log /root/."
sshpass -p "ChangeMe" scp $ric:/root/xapp-logger.log ./$out_dir/

sshpass -p "ChangeMe" ssh $ric "rm /root/xapp-logger.log"
