#!/bin/bash

LISTDS="Caltech101
DescribableTextures
EuroSAT
FGVCAircraft
Food101
OxfordFlowers
OxfordPets
SUN397
StanfordCars
UCF101
PLANTDOC
CUB
StanfordDogs
ImageNet
"


ATTEMPT=$1
SEED=$2
MODEL=$3
AUGMENT=$4

echo ${MODTYPE}
for ds in $LISTDS;
do
    DIR=/app/few_shot_final/output/results${ATTEMPT}/seed_${SEED}/
    if [ -e "${DIR}results_${ds}.pkl" ]; then
        echo "Oops! The results exist at '${DIR}results_${ds}.pkl' (so skip this job)"
    else
        echo "Saving dir ${DIR}"
        echo "Attempt ${ATTEMPT} DS ${ds} seed ${SEED} model ${MODEL}"
        python3 run.py \
        --datasetname ${ds} \
        --seed ${SEED} \
        --num_augment ${AUGMENT} \
        --model_path /app/few_shot_final/${MODEL} \
        --save_dir ${DIR}
    fi
done
