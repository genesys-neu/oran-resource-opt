#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 8 UEs (if you have less than 8, use 'genesys-' as a place holder
## Then there should be the interferer
## Then there should be the observer
## Then there should be the near-RT RIC

#set -x


out_dir=${1%.*}
num=1010123456002


echo "using $1, results will be saved in ./$out_dir"
sleep 5
mkdir $out_dir

read -r gnb < $1
echo "gnb is: $gnb"
# Start the channel
echo "Starting the channel emulator"
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'
sleep 10

ric=$(sed '12!d' $1)

# start the gNB and UEs
for num in $(seq 1 11); do
    line=$(sed -n "${num}p" "$1")
    echo "Starting SRN: $line"
    sshpass -p "scope" scp radio_rb.conf $line:/root/radio_api/radio_rb.conf
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_rb.conf" &
    sshpass -p "scope" rsync -avz ./raw/ $line:/root/TRACTOR/raw
    if [ $line = $gnb ]
    then
      echo "Copying new csv_reader.c to gNB"
      sleep 2
      sshpass -p "scope" rsync -avz colosseum-scope-e2/src/du_app/ $gnb:/root/radio_code/colosseum-scope-e2/src/du_app/
      sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/src/du_app/ && g++ readLastMetrics.cpp -o readLastMetrics.o"
      echo "Letting the gNB start"
      sleep 30
    fi
    sleep 2
    clear -x
done

#exit 0

echo "Setting up the near-RT RIC"
# TODO: Update the line below with the correct xapp python file
sshpass -p "ChangeMe" scp run_xapp_multi_obj.py $ric:/root/TRACTOR/run_xapp_IMPACT.py
sshpass -p "ChangeMe" rsync -avz ./utils/ $ric:/root/TRACTOR/utils/
sshpass -p "ChangeMe" rsync -avz ./model/ $ric:/root/TRACTOR/model/

sshpass -p "ChangeMe" ssh $ric 'cd ~ && cd radio_code/colosseum-near-rt-ric/setup-scripts/ && ./setup-ric.sh col0'
sleep 15

# connect the gNB and RIC
IPCOL0=`sshpass -p "ChangeMe" ssh $ric 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo "The IP adderss for the near-RT RIC is: $IPCOL0"

#exit 0

echo "Setting up the E2 interface on the gNB"
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && sed -i 's/172.30.105.104/${IPCOL0}/' build_odu.sh && ./build_odu.sh clean" # && ./run_odu.sh

sleep 20
clear -x
# Start the ODU
echo "Starting the ODU"
#gnome-terminal -- bash -c "sshpass -p 'scope' ssh -t $gnb 'cd /root/radio_code/colosseum-scope-e2/ && sh run_odu.sh'; bash" &
#gnome-terminal -- bash -c "sshpass -p 'scope' ssh -t $gnb 'cd /root/radio_code/colosseum-scope-e2/ && sh run_odu.sh' | tee ./odulogfile.log; bash" &
gnome-terminal -- bash -c "sshpass -p 'scope' ssh -t $gnb 'cd /root/radio_code/colosseum-scope-e2/ && sh run_odu.sh' | awk '/Received RIC control message/ {print; for(i=1;i<=20;i++) {getline; print}}' | tee ./odulogfile.log; bash" &


sleep 20

echo "Starting the Near-RT RIC"
GNBID=`sshpass -p "ChangeMe" ssh $ric "docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1"`
echo "The gnb ID is: $GNBID"
sshpass -p "ChangeMe" ssh $ric "cd /root/radio_code/colosseum-near-rt-ric/setup-scripts && ./setup-sample-xapp.sh ${GNBID}"

sleep 15
clear -x

echo "Copying files to the xApp"
sshpass -p "ChangeMe" ssh $ric 'docker cp /root/TRACTOR sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $ric 'docker exec sample-xapp-24 mv /home/sample-xapp/TRACTOR/utils/run_xapp_IMPACT.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp_IMPACT.sh'

echo "Starting the xApp"
gnome-terminal -- bash -c "sshpass -p 'ChangeMe' ssh $ric 'docker exec -i sample-xapp-24 bash -c \"rm /home/*.log && cd /home/sample-xapp/ && ./run_xapp_IMPACT.sh\"'; bash" &

sleep 30
clear -x
echo "Configured all SRNs"
sleep 30

# remove any existing metrics
echo "Removing any existing metrics"
sshpass -p "scope" ssh $gnb "
for file in /root/radio_code/scope_config/metrics/csv/101*_metrics.csv; do
    sed -i '2,\$d' \"\$file\"
done
"

# call the traffic script
bash ./run_traffic.sh $1
