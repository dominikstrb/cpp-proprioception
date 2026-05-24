#!/bin/bash

participants=(
    135033060
    173058413
    177196795
    182095555
    212056115
    330954011
    524048600
    544725341
    644801403
    657093471
    714671738
    741917610
    765738475
    799755310
    802096684
    879521247
    891444214
    910949717
    918338248
)

for participant in "${participants[@]}"; do
    source .venv/bin/activate
    echo "Fitting participant $participant"
    nohup python fit_multisensory.py --participant $participant --model optimal --model_class MultisensoryDelayModelPointMassDynamics --seed 7453 --nwarmup 1000 --nsamp 2500 > logs/$participant-optimal-MultisensoryDelayModelPointMassDynamics.out 2>&1 &
    nohup python fit_multisensory.py --participant $participant --model no_integration --model_class MultisensoryDelayModelPointMassDynamics --seed 7453 --nwarmup 1000 --nsamp 2500 > logs/$participant-no_integration-MultisensoryDelayModelPointMassDynamics.out 2>&1 &
    nohup python fit_multisensory.py --participant $participant --model equal_integration --model_class MultisensoryDelayModelPointMassDynamics --seed 7453 --nwarmup 1000 --nsamp 2500 > logs/$participant-equal_integration-MultisensoryDelayModelPointMassDynamics.out 2>&1 &
    wait
    echo "Finished fitting participant $participant"
done
