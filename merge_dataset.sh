#!/bin/bash
mkdir -p /home/ibroto/Documents/SMC/SPIS/BodyInTransit/DATASET/train
mkdir -p /home/ibroto/Documents/SMC/SPIS/BodyInTransit/DATASET/test

for i in $(seq -w 1 12); do
    ZIPFILE="/home/ibroto/Documents/SMC/SPIS/BodyInTransit/MLSP_bit_dataset-20250429T072910Z-0$i.zip"
    TEMPDIR="/home/ibroto/Documents/SMC/SPIS/BodyInTransit/temp_unzip"

    echo "Unzipping $ZIPFILE..."
    unzip -q "$ZIPFILE" -d "$TEMPDIR"

    for subset in train test; do
        SUBSET_SRC=$(find "$TEMPDIR" -type d -name "$subset" | head -n 1)

        if [ -n "$SUBSET_SRC" ]; then
            echo "Merging $subset from $ZIPFILE..."
            rsync -a --ignore-existing "$SUBSET_SRC/" "DATASET/$subset/"
        else
            echo "Warning: '$subset' not found in $ZIPFILE"
        fi
    done

    rm -rf "$TEMPDIR"
done

