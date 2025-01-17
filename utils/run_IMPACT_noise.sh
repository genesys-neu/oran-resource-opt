#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt
## Ensure the gNB is the first SRN in config_file.txt
## Then there should be 8 UEs
## Then there should be the interferer
## Then there should be the observer
## Then there should be the near-RT RIC

#set -x

eNB_PORT=5305
UE_PORT=5415
ip=3
out_dir=${1%.*}
num=1010123456002


echo "using $1, results will be saved in ./$out_dir"
sleep 10
mkdir $out_dir

read -r gnb < $1
echo "gnb is: $gnb"
# Start the channel
sshpass -p "scope" ssh $gnb 'colosseumcli rf start 10042 -c'


interferer=$(sed '10!d' $1)
sshpass -p "sunflower" scp uhd_tx_tone.sh $interferer:/root/utils/

listener=$(sed '11!d' $1)

ric=$(sed '12!d' $1)
#
# start the gNB and UEs
for num in $(seq 1 9); do
    line=$(sed -n "${num}p" "$1")
    echo "Configuring SRN: $line"
    sshpass -p "scope" rsync -av -e ssh --exclude '.git' ../colosseum-scope/radio_code/. $line:/root/radio_code/.
    sshpass -p "scope" ssh $line "cd /root/radio_code/srsLTE && rm -rf build && mkdir build && cd build && cmake .."
    sshpass -p "scope" ssh $line "cd /root/radio_code/srsLTE/build && make -j12 && sudo make install && srslte_install_configs.sh user" &
    sleep 5
    clear -x
done

sleep 180
echo "All srs Nodes built"

for num in $(seq 1 9); do
    line=$(sed -n "${num}p" "$1")
    echo "Starting SRN: $line"
    #To enable IMPACT use the following line:
    sshpass -p "scope" scp radio_IMPACT_2.conf $line:/root/radio_api/
    sshpass -p "scope" ssh $line "cd /root/radio_api && python3 scope_start.py --config-file radio_IMPACT_2.conf" &
    sshpass -p "scope" rsync -av -e ssh --exclude 'colosseum*' --exclude '.git' --exclude 'logs' --exclude 'utils' --exclude 'model' ../../TRACTOR $line:/root/.
    if [ $line = $gnb ]
    then
      echo "Letting the gNB start"
      sleep 15
    fi
    sleep 2
    clear -x
done

#exit 0

echo "Setting up the near-RT RIC"
sshpass -p "ChangeMe" ssh $ric 'cd ~ && cd radio_code/colosseum-near-rt-ric/setup-scripts/ && ./setup-ric.sh col0'
sleep 15

# connect the gNB and RIC
IPCOL0=`sshpass -p "ChangeMe" ssh $ric 'ifconfig col0 | grep '"'"'inet addr'"'"' | cut -d: -f2 | awk '"'"'{print $1}'"'"''`
echo "The IP adderss for the near-RT RIC is: $IPCOL0"

#exit 0

echo "Setting up the E2 interface on the gNB"
sshpass -p "scope" rsync -av -e ssh --exclude '.git' ../colosseum-scope-e2/. $gnb:/root/radio_code/colosseum-scope-e2/.
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/src/du_app/ && g++ readLastMetrics.cpp -o readLastMetrics.o"
sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && sed -i 's/172.30.105.104/${IPCOL0}/' build_odu.sh && ./build_odu.sh clean" # && ./run_odu.sh

sleep 20
clear -x
# Start the ODU
echo "Starting the ODU"
gnome-terminal -- bash -c "sshpass -p 'scope' ssh -t $gnb 'cd /root/radio_code/colosseum-scope-e2/ && sh run_odu.sh'; bash" &
#sshpass -p "scope" ssh $gnb "cd /root/radio_code/colosseum-scope-e2/ && ./run_odu.sh" &

sleep 20

echo "Starting the Near-RT RIC"
GNBID=`sshpass -p "ChangeMe" ssh $ric "docker logs e2term | grep -Eo 'gnb:[0-9]+-[0-9]+-[0-9]+' | tail -1"`
echo "The gnb ID is: $GNBID"
sshpass -p "ChangeMe" ssh $ric "cd /root/radio_code/colosseum-near-rt-ric/setup-scripts && ./setup-sample-xapp.sh ${GNBID}"

sleep 15
clear -x

echo "Copying files to the xApp"
sshpass -p "ChangeMe" rsync -av -e ssh --exclude 'colosseum*' --exclude '.git' --exclude 'logs/*UE/' --exclude 'utils/raw' --exclude 'raw' ../../TRACTOR $ric:/root/.
sshpass -p "ChangeMe" ssh $ric 'docker cp /root/TRACTOR sample-xapp-24:/home/sample-xapp/.'
sshpass -p "ChangeMe" ssh $ric 'docker exec sample-xapp-24 mv /home/sample-xapp/TRACTOR/utils/run_xapp_IMPACT.sh /home/sample-xapp/. && docker exec sample-xapp-24 chmod +x /home/sample-xapp/run_xapp_IMPACT.sh'

echo "Starting the xApp"
#sshpass -p "ChangeMe" ssh $ric 'docker exec -i sample-xapp-24 bash -c "rm /home/*.log && cd /home/sample-xapp/ && ./run_xapp_IMPACT.sh"' &
gnome-terminal -- bash -c "sshpass -p 'ChangeMe' ssh $ric 'docker exec -i sample-xapp-24 bash -c \"rm /home/*.log && cd /home/sample-xapp/ && ./run_xapp_IMPACT.sh\"'; bash" &

sleep 20
echo "Starting the listener"
sshpass -p "sunflower" ssh $listener "sed -i 's/--freq 1\.010e9 --rate 1e6/--freq 1.020e9 --rate 2e7/' utils/uhd_rx_fft.sh"
gnome-terminal -- bash -c "sshpass -p 'sunflower' ssh -t $listener 'sh utils/uhd_rx_fft.sh'; bash" &

sleep 30
clear -x
echo "Configured all SRNs"
sleep 30

# remove any existing metrics
sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"

# call the traffic script
if [ $(wc -l < "$1") -ge 10 ]; then
  sh ./run_traffic_interf.sh $gnb $(sed -n "2p" "$1") $interferer
  echo "starting traffic gen with interference"
else
  sh ./run_traffic_interf.sh $gnb $(sed -n "2p" "$1")
  echo "starting traffic gen"
fi

echo "All tests complete"
kill $(jobs -p)
