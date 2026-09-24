# Brazilian Semantic BERT

Training and evaluation of Transformer models for **Semantic Role Labeling
(SRL) in Brazilian Portuguese**, using PropBank.Br data. Each instance
represents a sentence with a known predicate marked by special tokens, and the
model classifies semantic roles at the token level.

The project supports comparisons among different encoders, loss strategies,
and a specialist ensemble. It also provides confusion matrices, per-role F1
comparisons, and a dashboard for qualitative analysis with linguists.

## Models and strategies

| Alias | Versions | Hugging Face model |
|---|---|---|
| `bertimbau` | `base`, `large` | BERTimbau |
| `bert-multilingual` | `base` | BERT multilingual cased |
| `xlm-roberta` | `base`, `large` | XLM-RoBERTa |
| `norberto` | `base`, `large` | NorBERTo |

Available strategies:

- `baseline`: cross-entropy loss;
- `focal_loss`: focal loss with $\gamma = 2$;
- `specialists_ensemble`: separate specialists for numbered arguments and
	modifiers, merged by confidence.

Global precision, recall, and F1 are macro-averaged, excluding the `O` and
`PRED` classes.

## Project structure

```text
data/raw/                  original data
data/processed/            processed splits and label vocabulary
src/configs/               models associated with each alias
src/data/                  dataset parsing, construction, and splitting
src/features/              tokenization and label alignment
src/pipelines/             training, evaluation, inference, and ensemble
src/training/              metrics and training components
src/utils/                 analyses, plots, and dashboards
artifacts/                 generated models, metrics, and results
mlruns/                    artifacts managed by MLflow
```

## Environment

The recommended environment uses Docker and an NVIDIA GPU. From the project
root:

```bash
docker build --no-cache --pull -t brazilian_semantic_bert .

docker run --gpus all -it \
	--name brazilian_semantic_bert \
	-v "$(pwd):/app" \
	-w /app \
	-p 5000:5000 \
	-p 8765:8765 \
	brazilian_semantic_bert
```

To return to an existing container:

```bash
docker start -ai brazilian_semantic_bert
```

Run the Python commands in this document inside the container from `/app`.
Dependencies are pinned in `requirements.txt`.

## Training

The training interface receives six positional arguments, all prefixed with
`--`:

```text
--model --version --epochs --batch-size --strategy --seed
```

Example with two GPUs:

```bash
torchrun --nproc_per_node=2 -m src.pipelines.train \
	--bertimbau --base --50 --256 --baseline --42
```

Example with focal loss:

```bash
torchrun --nproc_per_node=2 -m src.pipelines.train \
	--xlm-roberta --large --50 --128 --focal_loss --42
```

For a single GPU, use `--nproc_per_node=1`. Training uses early stopping,
saves the best checkpoint, and logs metrics to MLflow.

Results follow this convention:

```text
artifacts/<model>/<strategy>/seed<seed>/
├── final_model/
├── final_metrics.json
└── training_logs.txt
```

### Specialist ensemble

The orchestrator trains both specialists and merges their predictions:

```bash
bash orchestrator.sh bertimbau base 50 256 42
```

Artifacts are isolated under `specialists_ensemble` and do not overwrite
`baseline` or `focal_loss` results.

## MLflow tracking

```bash
python -m mlflow ui \
	--backend-store-uri sqlite:///mlflow.db \
	--host 0.0.0.0 \
	--port 5000
```

The interface is available at `http://127.0.0.1:5000` when the port is
published by Docker or forwarded through VS Code Remote SSH.

## Quantitative analysis

### Overall metrics table

Aggregates precision, recall, and F1 by model and strategy. When multiple seeds
are available, it reports the mean and standard deviation.

```bash
python -m src.utils.analyze_metrics_table --artifacts-dir artifacts
```

To restrict the comparison to specific strategies:

```bash
python -m src.utils.analyze_metrics_table \
	--artifacts-dir artifacts \
	--strategies baseline focal_loss
```

Outputs in `artifacts/comparisons/metrics_table/`:

- `model_metrics.png`: table rendered as an image;
- `model_metrics.csv`: tabular data;
- `model_metrics.tex`: LaTeX table using `booktabs`, ready for Overleaf.

### F1 by semantic role

```bash
python -m src.utils.analyze_f1_comparison \
	--artifacts-dir artifacts \
	--data-dir data/processed \
	--strategy baseline
```

CSV, PNG, and interactive HTML versions are generated under
`artifacts/comparisons/<strategy>/`.

### Confusion matrices

```bash
python -m src.utils.analyze_confusion \
	--model-path artifacts/xlm-roberta-large/baseline/seed120/final_model \
	--base-model xlm-roberta-large
```

The `confusion_analysis/` directory is created next to the model and contains
confusion matrices, metrics, token-level predictions, and
`instance_analysis.json`.

## Qualitative analysis

First, select representative examples from the instance-level analysis:

```bash
python -m src.utils.select_qualitative_examples \
	--input artifacts/xlm-roberta-large/baseline/seed120/confusion_analysis/instance_analysis.json \
	--count 5
```

The `qualitative_examples.json` file contains three groups with distinct
sentences:

- correctly classified numbered arguments (`ARG0`, `ARG1`, ...);
- correctly classified modifiers (`ARGM-*`);
- total failures when classifying true arguments.

### Linguistic inspection dashboard

```bash
python -m src.utils.build_qualitative_dashboard
```

By default, the dashboard uses the examples from
`artifacts/xlm-roberta-large/baseline/seed120/confusion_analysis/` and creates
a self-contained HTML file at:

```text
artifacts/qualitative_dashboard/index.html
```

You can provide another selection explicitly:

```bash
python -m src.utils.build_qualitative_dashboard \
	--input artifacts/<model>/<strategy>/seed<seed>/confusion_analysis/qualitative_examples.json
```

The dashboard provides search, category and role filters, predicate
highlighting, lemma and frame information, identification metrics, confidence
scores, and gold-versus-prediction comparisons.

To serve it locally, run this command from the project root:

```bash
python3 -m http.server 8765 --bind 0.0.0.0
```

Then open:

```text
http://127.0.0.1:8765/artifacts/qualitative_dashboard/index.html
```

In a Remote SSH session, forward port `8765` from the VS Code **PORTS** panel.
You can also open the HTML file directly from the editor's Explorer.