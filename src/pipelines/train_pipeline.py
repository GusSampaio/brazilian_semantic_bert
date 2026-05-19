import os
import torch
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from data.srl_data_module import SRLDataModule
from training.metrics import SRLMetrics

def main():
    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        data_path="../data/processed/",
        use_preprocessed_data=True,
        model_name="neuralmind/bert-base-portuguese-cased",
        predicate_signal="special_token",
    )
    
    hf_datasets = data_module.datasets
    train_dataset = hf_datasets["train"]
    dev_dataset = hf_datasets["validation"]
    
    model = AutoModelForTokenClassification.from_pretrained(
        data_module.model_name,
        num_labels=len(data_module.label2id),
        id2label=data_module.id2label,
        label2id=data_module.label2id
    )

    model.resize_token_embeddings(len(data_module.tokenizer.tokenizer))
 
    data_collator = DataCollatorForTokenClassification(
        tokenizer=data_module.tokenizer.tokenizer,
        padding=True # dynamic padding 
    )

    metrics_calculator = SRLMetrics(id2label=data_module.id2label)

    training_args = TrainingArguments(
        output_dir="artifacts/srl_model_checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # Use of mixed precision if using GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics
    )

    print("Starting training...")
    trainer.train()

    print("Saving best model at artifacts/srl_final_model...")
    trainer.save_model("artifacts/srl_final_model")
    print("End of training!")

if __name__ == "__main__":
    main()