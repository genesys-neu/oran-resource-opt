#!/bin/bash
## HOW TO RUN: ./interference_tgen.sh genesys-gNB genesys-UE genesys-Interferance


if [ ! -d "./interference" ]
  then
    mkdir ./interference
    mkdir ./interference/off
    mkdir ./interference/on
fi


if [ $# -eq 3 ]
  then
    echo "Starting interference"
    sshpass -p "sunflower" ssh $3 "cd /root/utils && sh uhd_tx_tone.sh" &
    sleep 5 # let the tx process start
    int_PID=`sshpass -p "sunflower" ssh $3 "pgrep tx_waveforms"`
    echo "****** Returned PID: ${int_PID} ***********"
fi

for t in ../raw/*.csv
do
  sshpass -p "scope" ssh $1 "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"
  tracename=$(basename ${t})
  echo "TRACE ${tracename}"
    # Check if tracename variable is not empty before extracting substring
  if [ -n "$tracename" ]; then
      # Extract the first character of tracename
      first_char=$(echo "${tracename}" | cut -c1)
      # Perform actions based on the value of first_char
      if [ "$first_char" = "e" ]; then
          sshpass -p "scope" ssh "$1" "sed -i '2s/::[0-9]/::2/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt"
          echo "Setting slice to eMBB"
      elif [ "$first_char" = "u" ]; then
          sshpass -p "scope" ssh "$1" "sed -i '2s/::[0-9]/::1/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt"
          echo "Setting slice to URLLC"
      else
          sshpass -p "scope" ssh "$1" "sed -i '2s/::[0-9]/::0/' /root/radio_code/scope_config/slicing/ue_imsi_slice.txt"
          echo "Setting slice to mMTC"
      fi
  fi
  echo "***** Run traffic on gNB *****"
  # TODO: Remove ports from the following lines
  sshpass -p "scope" ssh -tt $1 "cd /root/TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} -eNBp 5306 -UEp 5416" &   # this will have to let the next command go through
  gNB_PID=$!
  echo "Sleep for 5 secs"
  sleep 5  # let's wait few seconds
  echo "***** Run traffic on UE *****"
  sshpass -p "scope" ssh -tt $2 "cd /root/TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp 5306 -UEp 5416" &
  UE_PID=$!
  sleep 5 # let the traffic start

  wait $gNB_PID # this will wait until gNB processes terminates
  wait $UE_PID # this will wait until gNB processes terminates

  echo "***** Sleeping... *****"
  sleep 5 # sleep for a few second to allow all the classifier outputs to complete producing files
  echo "***** Copy data *****"

  if [ $# -eq 3 ]
    then
      sshpass -p "scope" scp $1:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./interference/on/${tracename}
    else
      sshpass -p "scope" scp $1:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ./interference/off/${tracename}
  fi
  sshpass -p "scope" ssh $1 "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"

  echo "***** Completed $t Preparing for next run *****"
  sleep 5 # sleep for a few second to allow the system to settle
  clear -x
  
done


if [ $# -eq 3 ]
  then
    echo "***** Stopping Interference PID: ${int_PID} *****"
    sshpass -p "sunflower" ssh $3 "kill -INT ${int_PID}"
fi

echo "All tests complete"


