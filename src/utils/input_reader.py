import sys

def define_model():
    usage = """
Usage:
    python -m src.pipelines.train --<model> --<size>

Available models:
    --bertimbau          --base | --large
    --xlm-roberta        --base | --large
    --norberto           --base | --large
    --bert-multilingual  --base

Examples:
    python -m src.pipelines.train --bertimbau --base
    python -m src.pipelines.train --xlm-roberta --large
"""

    try:
        model_name = sys.argv[1].replace("--", "")
        model_size = sys.argv[2].replace("--", "")
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
        valid_sizes = ", ".join(sorted(valid_models[model_name]))
        raise ValueError(
            f"Invalid size '{model_size}' for model '{model_name}'. "
            f"Valid sizes: {valid_sizes}."
        )

    return model_name, model_size