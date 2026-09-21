import json
import os
os.environ["HF_HOME"] = "/app/.hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from src.data.srl_data_module import SRLDataModule
from src.training.metrics import SRLMetrics
from src.utils.input_reader import define_exp_config

EXPERIMENT_NAME = "srl-portuguese"


def predict_and_merge(model_num, model_mod, dataset, data_collator, label2id, device):
    model_num.eval()
    model_mod.eval()

    valid_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    cols_to_keep = [col for col in dataset.column_names if col in valid_cols]
    eval_dataset = dataset.select_columns(cols_to_keep)

    max_seq_len = max(len(ex["input_ids"]) for ex in eval_dataset)
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)

    numbered_id2label = {
        int(label_id): label for label_id, label in model_num.config.id2label.items()
    }
    modifier_id2label = {
        int(label_id): label for label_id, label in model_mod.config.id2label.items()
    }
    unknown_labels = (
        set(numbered_id2label.values()) | set(modifier_id2label.values())
    ) - set(label2id)
    if unknown_labels:
        raise ValueError(
            f"Specialist models contain labels absent from the global vocabulary: "
            f"{sorted(unknown_labels)}"
        )

    numbered_o_id = model_num.config.label2id["O"]
    modifier_o_id = model_mod.config.label2id["O"]
    global_o_id = label2id["O"]
    num_labels = len(label2id)

    all_final_probs = []
    all_true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            logits_num = model_num(**inputs).logits
            logits_mod = model_mod(**inputs).logits

            probs_num = F.softmax(logits_num, dim=-1)
            probs_mod = F.softmax(logits_mod, dim=-1)

            preds_num = torch.argmax(probs_num, dim=-1)
            preds_mod = torch.argmax(probs_mod, dim=-1)

            merged_preds = torch.full_like(preds_num, fill_value=global_o_id)

            for b in range(preds_num.size(0)):
                for s in range(preds_num.size(1)):
                    p_n = preds_num[b, s].item()
                    p_m = preds_mod[b, s].item()
                    label_n = numbered_id2label[p_n]
                    label_m = modifier_id2label[p_m]

                    if p_n != numbered_o_id and p_m == modifier_o_id:
                        merged_preds[b, s] = label2id[label_n]
                    elif p_m != modifier_o_id and p_n == numbered_o_id:
                        merged_preds[b, s] = label2id[label_m]
                    elif p_n != numbered_o_id and p_m != modifier_o_id:
                        score_n = probs_num[b, s, p_n].item()
                        score_m = probs_mod[b, s, p_m].item()
                        selected_label = label_n if score_n >= score_m else label_m
                        merged_preds[b, s] = label2id[selected_label]

            merged_probs = F.one_hot(merged_preds, num_classes=num_labels).float()

            pad_len = max_seq_len - merged_probs.size(1)
            if pad_len > 0:
                merged_probs = F.pad(merged_probs, (0, 0, 0, pad_len), value=0.0)
                labels = F.pad(labels, (0, pad_len), value=-100)

            all_final_probs.append(merged_probs.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())

    return np.concatenate(all_final_probs, axis=0), np.concatenate(all_true_labels, axis=0)


def main(model_name, strategy, seed, num_epochs, early_stopping_patience):
    if strategy != "specialists_ensemble":
        raise ValueError(
            "Ensemble results must use strategy 'specialists_ensemble' to avoid "
            "overwriting metrics from single-model strategies."
        )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    output_path = f"artifacts/{model_name.split('/')[-1]}/{strategy}/seed{seed}"
    run_base_name = f"{model_name.split('/')[-1]}_{strategy}_seed{seed}"

    data_module = SRLDataModule(
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name=model_name,
        predicate_signal="special_token",
    )
    label2id = data_module.label2id
    id2label = data_module.id2label

    data_collator = DataCollatorForTokenClassification(tokenizer=data_module.tokenizer.tokenizer, padding=True)
    metrics_calculator = SRLMetrics(id2label=id2label, eval_mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading trained specialist models...")
    model_num = AutoModelForTokenClassification.from_pretrained(f"{output_path}/model_numbered").to(device)
    model_mod = AutoModelForTokenClassification.from_pretrained(f"{output_path}/model_modifiers").to(device)

    print("Running ensemble inference on test set...")
    preds, true_labels = predict_and_merge(
        model_num=model_num, model_mod=model_mod,
        dataset=data_module.datasets["test"], data_collator=data_collator,
        label2id=label2id, device=device
    )
    test_metrics = {
        f"test_{key}": value
        for key, value in metrics_calculator.compute_metrics((preds, true_labels)).items()
    }

    mlflow.start_run(run_name=f"{run_base_name}_ensemble")
    mlflow.set_tags({"component": "ensemble", "strategy": strategy})
    mlflow.log_params({
        "model_name": model_name, "strategy": "specialists_ensemble", "seed": seed,
        "num_epochs_ceiling": num_epochs, "early_stopping_patience": early_stopping_patience,
    })
    mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})

    metrics_path = f"{output_path}/final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
    mlflow.log_artifact(metrics_path)
    mlflow.end_run()

    print("\n================ MÉTRICAS DE TESTE (UNIFICADAS) ================")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    model_name, model_size, num_epochs, batch_size, strategy, seed = define_exp_config()
    cfg_path = f"src/configs/{model_name}.json"
    cfg = json.load(open(cfg_path))
    cfg = cfg[model_size]

    main(cfg["model_name"], strategy=strategy, seed=seed, num_epochs=num_epochs,
         early_stopping_patience=EARLY_STOPPING_PATIENCE if False else 10)