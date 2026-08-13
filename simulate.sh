#!/bin/bash

participants=(
    135033060
    # 173058413
    177196795
    182095555
    212056115
    # 330954011
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

model_classes=(
    "BoundedActorPointMassDynamics"
    "BoundedActor"
)

source .venv/bin/activate

for participant in "${participants[@]}"; do
    for model_class in "${model_classes[@]}"; do
        echo "Simulating participant $participant, model class $model_class"
        nohup python simulate_multisensory.py --participant $participant --seed 3 --nwarmup 2500 --model_class $model_class --conditions prop vis multi > logs/simulate-$participant-$model_class.log 2>&1 &
        wait
    done
done