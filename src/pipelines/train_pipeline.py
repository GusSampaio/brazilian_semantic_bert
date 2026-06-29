import os
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback
)

from src.data.srl_data_module import SRLDataModule
from src.training.metrics import SRLMetrics
from src.utils.json_loader import load_configs
from src.utils.input_reader import define_model

class TextLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("          LOGS DE TREINAMENTO DO MODELO            \n")
            f.write("==================================================\n\n")

    def on_log(self, args, state, control, logs=None, **kwargs):    
        """Disparado toda vez que o modelo loga perda ou métricas de validação"""
        if logs:
            with open(self.log_path, "a", encoding="utf-8") as f:
                # Formata a época e o passo atual de forma limpa
                prefixo = f"[Época {state.epoch:.2f} / Passo {state.global_step}] "
                
                # Converte o dicionário de métricas para uma string legível
                metricas_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
                
                f.write(prefixo + metricas_str + "\n")

def main(model_name, num_epochs, batch_size):
    data_module = SRLDataModule(
        data_path="data/processed/",
        use_preprocessed_data=True,
        model_name=model_name,
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
    
    output_path = f"artifacts/{model_name}"
    log_file_path = f"{output_path}/training_logs.txt"
    training_args = TrainingArguments(
        output_dir=f"{output_path}/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        fp16=torch.cuda.is_available(), # Use of mixed precision if using GPU
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        
        learning_rate=3e-5,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics, 
        callbacks=[TextLoggerCallback(log_file_path)]
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(f"{output_path}/final_model")
    print(f"End of training! Saved at {output_path}/final_model")
    print(f"Logs saved at: {log_file_path}")

    metrics = trainer.evaluate(dev_dataset)
    print(metrics)

if __name__ == "__main__":
    model_name, model_size = define_model()
    cfg_path = f"src/configs/{model_name}.json"
    cfg = load_configs(cfg_path)
    cfg = cfg[model_size]

    main(cfg["model_name"], num_epochs=50, batch_size=256)