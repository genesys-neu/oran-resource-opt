#!/bin/bash
## HOW TO RUN: .sh slice_testing.sh config_file.txt radio_slice.conf
## Ensure the gNB is the first SRN in config_file.txt

#set -x

eNB_PORT=5305
UE_PORT=5315
ip=3
out_dir=${2%.*}
file_name=$(basename "$2")
num=1010123456002

echo "using $2, results will be saved in $out_dir"

sleep 10
mkdir $out_dir

gnb=$(sed '1!d' $1)
echo "gnb is: $gnb"
ue=$(sed '2!d' $1)
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'

for num in $(seq 1 2); do
    line=$(sed -n "${num}p" "$1")
    echo "Configuring SRN: $line"
    sshpass -p "scope" scp $2 $line:/root/radio_api/
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file $file_name" &
    sshpass -p "scope" rsync -av -e ssh --exclude 'colosseum*' --exclude '.git' --exclude 'logs' --exclude 'utils' --exclude 'model' ../TRACTOR $line:/root/.
    if [ $line = $gnb ]
    then
      echo "Letting the gNB start"
      sleep 15
    fi
    sleep 2
    clear -x
done

sleep 30
clear -x
echo "Configured all SRNs"
sleep 30
sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"

for t in ./raw/*.csv
do
  tracename=$(basename "$t")
  if [ "$tracename" = "IMPACT.csv" ] || [ "$tracename" = "README.md" ]; then
    continue
  fi
  echo "TRACE DIR $t"
  echo "***** Run traffic on gNB *****"
  sshpass -p "scope" ssh -tt $gnb "cd /root/TRACTOR && python traffic_gen.py --eNB -f ${t}" &   # this will have to let the next command go through
  echo "Sleep for 5 secs"
  sleep 5  # let's wait few seconds
  echo "***** Run traffic on UE *****"
  start_ts=`date +%s%N | cut -b1-13`
  sshpass -p "scope" ssh -tt $ue "cd /root/TRACTOR && python traffic_gen.py -f ${t}" &
  wait  # this will wait until all child processes terminates
  end_ts=`date +%s%N | cut -b1-13`
  echo "START: $start_ts\nEND: $end_ts\n"
  #echo "$start_ts,$end_ts" > ${t}_se_info.out
  echo "***** Sleeping... *****"
  sleep 5 # sleep for a few second to allow all the classifier outputs to complete producing files
  echo "***** Copy data *****"

  sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/101*_metrics.csv ${out_dir}/${tracename}_metrics.csv
  clear -x
done


echo "All tests complete"
