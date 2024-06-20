#!/bin/bash

for i in data_v2/*/*.tif;
do
    bn=$(basename "$i" .tif)
    bd=$(dirname "$i")
    convert "$i" "$bd/$bn.png"
    echo $i
done