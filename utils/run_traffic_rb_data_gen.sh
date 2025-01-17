#!/bin/bash
## HOW TO RUN: ./setup_tgen.sh config_file.txt


# set -x

eNB_PORT=5305
UE_PORT=5415
ip=4
out_dir=${1%.*}
num=1010123456002

read -r gnb < $1
echo "gnb is: $gnb"


echo "Starting TGEN for eMBB 1"
tracename=embb_11_18.csv
#ip=4
echo "Using trace: ${tracename}"
line=$(sed '3!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for eMBB 2"
tracename=embb_03_03a.csv
#ip=5
echo "Using trace: ${tracename}"
line=$(sed '4!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for eMBB 3"
tracename=embb_04_10.csv
#ip=6
echo "Using trace: ${tracename}"
line=$(sed '5!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for eMBB 4"
tracename=embb_06_09.csv
#ip=7
echo "Using trace: ${tracename}"
line=$(sed '6!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for urllc 1"
tracename=urllc_05_18.csv
#ip=8
echo "Using trace: ${tracename}"
line=$(sed '7!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for urllc 2"
tracename=urllc_06_12.csv
#ip=9
echo "Using trace: ${tracename}"
line=$(sed '8!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))


echo "Starting TGEN for mmtc 1"
tracename=mmtc_05_18.csv
#ip=10
echo "Using trace: ${tracename}"
line=$(sed '9!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
ip=$((ip+1))
eNB_PORT=$((eNB_PORT+1))
UE_PORT=$((UE_PORT+1))

# remove any existing metrics
sshpass -p "scope" ssh $gnb "rm /root/radio_code/scope_config/metrics/csv/101*_metrics.csv"


echo "Starting TGEN for demo UE"
ip=3
tracename=$2
echo "Using trace: ${tracename}"
line=$(sed '2!d' $1)
echo "Using ip: 172.16.0.${ip}"
echo "Using eNB_PORT: $eNB_PORT and UE_PORT: $UE_PORT"
echo "Starting gNB"
sshpass -p "scope" ssh $gnb "cd TRACTOR && python traffic_gen.py --eNB -f ./raw/${tracename} --ip 172.16.0.${ip} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &
sleep 2
echo "Starting UE"
sshpass -p "scope" ssh $line "cd TRACTOR && python traffic_gen.py -f ./raw/${tracename} -eNBp ${eNB_PORT} -UEp ${UE_PORT}" &


sleep 60

echo "All traffic complete"
kill $(jobs -p)

#copy log files from the gNB
sshpass -p "scope" scp $gnb:/root/radio_code/scope_config/metrics/csv/*_metrics.csv ./$out_dir/

#copy log file from xApp
sshpass -p "ChangeMe" ssh $ric "docker cp sample-xapp-24:/home/xapp-logger.log /root/."
sshpass -p "ChangeMe" scp $ric:/root/xapp-logger.log ./$out_dir/

sshpass -p "ChangeMe" ssh $ric "rm /root/xapp-logger.log"

##TODO: Add additional calls to traces here
