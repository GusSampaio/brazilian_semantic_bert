#!/bin/bash
set -e

MODEL_NAME=$1         # ex: bertimbau
MODEL_SIZE=$2         # ex: base
NUM_EPOCHS=$3         # ex: 50
BATCH_SIZE=$4         # ex: 256
SEED=$5               # ex: 42
STRATEGY="specialists_ensemble"

echo "=== Training specialist: numbered ==="
torchrun --nproc_per_node=2 -m src.pipelines.train_specialist \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED" --numbered

echo "=== Training specialist: modifiers ==="
torchrun --nproc_per_node=2 -m src.pipelines.train_specialist \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED" --modifiers

echo "=== Running ensemble ==="
python -m src.pipelines.run_ensemble \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED"

echo "Pipeline complete!"



