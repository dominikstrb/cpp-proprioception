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
    python simulate_multisensory.py --participant $participant --model_class MultisensoryDelayModelPointMassDynamics
done