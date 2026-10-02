import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


class SRLMetrics:
    def __init__(self, id2label, eval_mode=False):
        self.id2label = id2label
        self.eval_mode = eval_mode
    def _safe_get_label(self, idx):
        """
        Ensure that the index will be found, whether it is np.int64, int, or string.
        If the model predicts an index outside the vocabulary, return "O".
        """
        if int(idx) in self.id2label:
            return self.id2label[int(idx)]
        elif str(idx) in self.id2label:
            return self.id2label[str(idx)]
        return "O"

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=2)

        true_predictions = [
            [self._safe_get_label(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._safe_get_label(l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_true = [item for sublist in true_labels for item in sublist]
        flat_pred = [item for sublist in true_predictions for item in sublist]

        # Calculating macro metrics ignoring "O" class
        labels_to_evaluate = list(set(flat_true) - {"O"} - {"PRED"})

        metrics = {
            "precision": float(precision_score(flat_true, flat_pred, labels=labels_to_evaluate, average="macro", zero_division=0)),
            "recall": float(recall_score(flat_true, flat_pred, labels=labels_to_evaluate, average="macro", zero_division=0)),
            "f1": float(f1_score(flat_true, flat_pred, labels=labels_to_evaluate, average="macro", zero_division=0)),
        }

        if getattr(self, "eval_mode", False):
            report = classification_report(
                flat_true,
                flat_pred,
                labels=labels_to_evaluate,
                output_dict=True,
                zero_division=0
            )
            for role in labels_to_evaluate:
                if role in report:
                    metrics[f"f1_{role}"] = report[role]["f1-score"]

        return metrics