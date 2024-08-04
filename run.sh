#!/bin/bash

cd ./svfl || return
rm pipeline_main.log
python pipeline_main.py

#cd ../privacy_attack_during_inference || return
#rm pipeline_mc.log
#python pipeline_mc.py

cd ../ || return
python log_parser.py