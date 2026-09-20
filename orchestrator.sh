#!/bin/bash
set -e

MODEL_NAME=$1        # ex: bertimbau
MODEL_SIZE=$2         # ex: base
NUM_EPOCHS=$3         # ex: 50
BATCH_SIZE=$4         # ex: 256
STRATEGY=$5           # ex: baseline
SEED=$6                # ex: 42

echo "=== Treinando especialista: numbered ==="
torchrun --nproc_per_node=2 -m src.pipelines.train_specialist \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED" --numbered

echo "=== Treinando especialista: modifiers ==="
torchrun --nproc_per_node=2 -m src.pipelines.train_specialist \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED" --modifiers

echo "=== Rodando ensemble ==="
python -m src.pipelines.run_ensemble \
    --"$MODEL_NAME" --"$MODEL_SIZE" --"$NUM_EPOCHS" --"$BATCH_SIZE" --"$STRATEGY" --"$SEED"

echo "Pipeline completo!"