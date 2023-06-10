#!/bin/bash

DATASETS=(aloe art car century flowers garbage picnic pikachu pipe plant roses table)

for DATASET in "${DATASETS[@]}"; do
    export DATASET="$DATASET"
    export RENDER_FOLDER_POST="renders-postprocessed/${DATASET}"
    
    python scripts/launch_nerf.py metrics \
        --input-folder ${RENDER_FOLDER_POST} \
        --pseudo-gt-experiment-name ${DATASET}---nerfacto---pseudo-gt \
        "$@"
done
