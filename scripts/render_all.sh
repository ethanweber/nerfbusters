#!/bin/bash

DATASETS=(aloe art car century flowers garbage picnic pikachu pipe plant roses table)

for DATASET in "${DATASETS[@]}"; do
    export DATASET="$DATASET"
    export OUTPUT_FOLDER_POST="outputs-postprocessed/${DATASET}"
    export RENDER_FOLDER_POST="renders-postprocessed/${DATASET}"
    
    python scripts/launch_nerf.py render \
        --input-folder "$OUTPUT_FOLDER_POST" \
        --output-folder "$RENDER_FOLDER_POST" \
        --downscale-factor 2 \
        "$@"
done
