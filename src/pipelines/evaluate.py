from datasets import Dataset
import json
import numpy as np
import os
from pathlib import Path
import random
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.data.conllu_parser import PBP_parser
from src.data.instance_builder import SRLInstanceBuilder
from src.data.splitter import SRLSplitter
from src.data.srl_data_module import SRLDataModule
from src.training.metrics import SRLMetrics

np.random.seed(42)
os.environ["HF_HOME"] = "/app/.hf_cache"
random.seed(42)

def log_metrics(metrics: dict, output_dir: str, filename: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir/ filename

    metrics = {
        k: float(v) if isinstance(v, (np.floating, np.float32, np.float64))
        else int(v) if isinstance(v, (np.integer,))
        else v
        for k, v in metrics.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Metrics saved to {output_path}")

def main():
    model_path = "GusSampaio/bert-base-portuguese-cased-srl"
    dataset_path = "data/processed/"

    print("Loading dataset...")
    data_module = SRLDataModule(
        data_path="data/processed/",
        use_preprocessed_data=True,
        model_name=model_path,
        predicate_signal="special_token",
    )
    test_dataset = data_module.datasets["test"]
    print(f"Number of dataset examples: {len(test_dataset)}")

    print("Loading saved model...")
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=data_module.tokenizer.tokenizer,
        padding=True
    )

    metrics_calculator = SRLMetrics(
        id2label=data_module.id2label,
        eval_mode=True
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=64,
        ),
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics,
    )

    print("\nRunning evaluation...")

    predictions_output = trainer.predict(test_dataset=test_dataset)
    metrics = predictions_output.metrics

    model_name=model_path.split('/')[1]
    log_metrics(metrics=metrics, output_dir=f"artifacts/{model_name}", filename="evaluation_metrics.json")

if __name__ == "__main__":
    main()