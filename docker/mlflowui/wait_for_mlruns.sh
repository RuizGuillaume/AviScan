#!/bin/bash

while [ ! -d "./volume_data/mlruns/157975935045122495" ] 
do
    sleep 1
done

cd volume_data
sleep 5
mlflow ui --host 0.0.0.0 --port 5200