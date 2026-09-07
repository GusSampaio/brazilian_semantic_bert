import json
import os
import torch.distributed as dist
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.utils.input_reader import define_exp_config

EXPERIMENT_NAME = "srl-portuguese"
EARLY_STOPPING_PATIENCE=10

class TextLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("TRAINING LOGS\n")

    def on_log(self, args, state, control, logs=None, **kwargs):    
        """Triggered every time the model logs loss or validation metrics"""
        if logs:
            with open(self.log_path, "a", encoding="utf-8") as f:
                prefix = f"[Epoch {state.epoch:.2f} / Step {state.global_step}] "
                metrics = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
                f.write(prefix + metrics + "\n")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # squeezes the logits and targets to 2D and 1D respectively for loss computation
        # Logits [B, S, C] -> [B*S, C] | Targets [B, S] -> [B*S]
        num_classes = logits.size(-1)
        logits = logits.view(-1, num_classes)
        targets = targets.view(-1)

        # Filtering out the padding tokens (ignore_index) from the loss computation
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_pt = F.log_softmax(logits, dim=-1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()
    
class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_strategy="baseline", class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_strategy = loss_strategy
        self.class_weights = class_weights
        self.gamma = gamma

class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_strategy="baseline", class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_strategy = loss_strategy
        self.class_weights = class_weights
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Logits -> [batch_size * seq_len, num_labels]
        # Labels -> [batch_size * seq_len]
        num_labels = logits.size(-1)
        flat_logits = logits.view(-1, num_labels)
        flat_labels = labels.view(-1)

        if self.loss_strategy == "focal_loss":
            loss_fct = FocalLoss(
                gamma=self.gamma, 
                alpha=self.class_weights, 
                ignore_index=-100
            )
            loss = loss_fct(flat_logits, flat_labels)

        elif self.loss_strategy == "weighted_loss" and self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            loss = loss_fct(flat_logits, flat_labels)

        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(flat_logits, flat_labels)

        return (loss, outputs) if return_outputs else loss

def main(model_name, num_epochs, batch_size, strategy="baseline", seed=42, early_stopping_patience=EARLY_STOPPING_PATIENCE):
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

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        loss_strategy=strategy,
        class_weights=None,
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
    
    # Registering metrics and artifacts in MLflow only if this is the main process (local_rank == 0)
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
    cfg = json.load(open(cfg_path))
    cfg = cfg[model_size]

    main(
        cfg["model_name"],
        num_epochs=num_epochs,
        batch_size=batch_size,
        strategy=strategy,
        seed=seed,
    )