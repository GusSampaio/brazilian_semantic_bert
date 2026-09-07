import sys

def define_exp_config():
    usage = """
Usage:
    python -m src.pipelines.train --<model> --<version> --<num_epochs> --<batch_size> --<strategy> --<seed>

Available models:
    --bertimbau          --base | --large
    --xlm-roberta        --base | --large
    --norberto           --base | --large
    --bert-multilingual  --base

Examples:
    python -m src.pipelines.train --bertimbau --base --10 --32 --baseline --42
    python -m src.pipelines.train --xlm-roberta --large
"""

    try:
        model_name = sys.argv[1].replace("--", "")
        model_size = sys.argv[2].replace("--", "")
        num_epochs = int(sys.argv[3].replace("--", ""))
        batch_size = int(sys.argv[4].replace("--", ""))
        strategy = sys.argv[5].replace("--", "")
        seed = int(sys.argv[6].replace("--", ""))
    except IndexError:
        raise ValueError(
            "Missing required arguments.\n" + usage
        )

    valid_models = {
        "bertimbau": {"base", "large"},
        "xlm-roberta": {"base", "large"},
        "bert-multilingual": {"base", "large"},
        "norberto": {"base", "large"},
    }

    if model_name not in valid_models:
        raise ValueError(
            f"Unknown model '{model_name}'.\n\n{usage}"
        )

    if model_size not in valid_models[model_name]:
        valid_versions = ", ".join(sorted(valid_models[model_name]))
        raise ValueError(
            f"Invalid version '{model_size}' for model '{model_name}'. "
            f"Valid versions: {valid_versions}."
        )

    if num_epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

    if strategy not in {"baseline", "weighted_loss"}:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Valid strategies: baseline, weighted_loss."
        )

    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    

    return model_name, model_size, num_epochs, batch_size, strategy, seed