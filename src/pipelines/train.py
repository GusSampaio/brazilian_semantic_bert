import json
import os
import torch.distributed as dist
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import torch
import mlflow
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.integrations import MLflowCallback

from src.data.srl_data_module import SRLDataModule
from src.training.metrics import SRLMetrics
from src.utils.json_loader import load_configs
from src.utils.input_reader import define_exp_config

EXPERIMENT_NAME = "srl-portuguese"
EARLY_STOPPING_PATIENCE=10

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

def main(model_name, num_epochs, batch_size, strategy="baseline", seed=SEED, early_stopping_patience=EARLY_STOPPING_PATIENCE):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{model_name}_{strategy}_seed{seed}"

    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name=model_name,
        predicate_signal="special_token",
    )
    
    hf_datasets = data_module.datasets
    train_dataset, validation_dataset, test_dataset = hf_datasets["train"], hf_datasets["validation"], hf_datasets["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        data_module.model_name,
        num_labels=len(data_module.label2id),
        id2label=data_module.id2label,
        label2id=data_module.label2id
    )

    model.resize_token_embeddings(len(data_module.tokenizer.tokenizer),
                                  mean_resizing=True)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=data_module.tokenizer.tokenizer,
        padding=True # dynamic padding 
    )

    metrics_calculator = SRLMetrics(id2label=data_module.id2label)
    
    output_path = f"artifacts/{model_name.split('/')[-1]}/{strategy}/seed{seed}"
    log_file_path = f"{output_path}/training_logs.txt"

    run_name = f"{model_name.split('/')[-1]}_{strategy}_seed{seed}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))


    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,  # Avoids errors with DDP when using multiple GPUs
        run_name=run_name,

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
        logging_steps=10,

        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    if local_rank == 0:
        mlflow.start_run(run_name=run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=data_module.tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics, 
        callbacks=[TextLoggerCallback(log_file_path), 
                    EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                    MLflowCallback()]
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(f"{output_path}/final_model")
    print(f"End of training! Saved at {output_path}/final_model")

    trainer.pop_callback(EarlyStoppingCallback)

    metrics_calculator.eval_mode = True
    val_metrics = trainer.evaluate(validation_dataset, metric_key_prefix="best_val")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    
    # Registra parâmetros customizados na run ativada pelo Trainer
    if local_rank == 0:
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_params({
                "strategy": strategy,
                "num_epochs_ceiling": num_epochs,
                "early_stopping_patience": early_stopping_patience,
            })

            # Salva arquivos locais
            metrics_path = f"{output_path}/final_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({**val_metrics, **test_metrics}, f, indent=4)


            mlflow.log_artifact(log_file_path)
            mlflow.log_artifact(metrics_path)

            mlflow.end_run()

        print(f"Métricas finais (teste): {test_metrics}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    model_name, model_size, num_epochs, batch_size, strategy, seed = define_exp_config()
    cfg_path = f"src/configs/{model_name}.json"
    cfg = load_configs(cfg_path, model_size)

    main(
        cfg["model_name"],
        num_epochs=num_epochs,
        batch_size=batch_size,
        strategy=strategy,
        seed=seed,
    )