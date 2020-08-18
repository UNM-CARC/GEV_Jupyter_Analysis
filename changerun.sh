#!/bin/bash

orig=$1
new=$2

mv mlruns/${orig} mlruns/${new}

for d in $(find mlruns/${new} -maxdepth 1 -type d)
do
    echo ${d}
    sed "s|\/mlruns\/$orig|\/mlruns\/$new|g" ${d}/meta.yaml > ${d}/new_meta.yaml
    sed "s|experiment_id: '$orig'|experiment_id: '$new'|g" ${d}/new_meta.yaml > ${d}/meta.yaml
    rm ${d}/new_meta.yaml
done
