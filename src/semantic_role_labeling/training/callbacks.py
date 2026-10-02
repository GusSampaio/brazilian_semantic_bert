from pathlib import Path

from transformers import TrainerCallback


class TextLoggerCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("TRAINING LOGS\n", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        prefix = f"[Epoch {state.epoch:.2f} / Step {state.global_step}] "
        metrics = " | ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in logs.items()
        )
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(prefix + metrics + "\n")