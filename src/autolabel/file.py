from pathlib import Path
import urllib.request
import yaml


def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def download_models(model_dir: Path):
    info_path = model_dir / "info.yaml"
    with open(info_path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    print(
        f"[INFO] App: {info.get('app_name')} "
        f"v{info.get('app_version')}"
    )

    model_paths = {}
    models = info["model"]
    for model_type, model_info in models.items():
        name = model_info["name"]
        download_link = model_info["download_link"]
        model_path = model_dir / name
        weight_path = model_path / "weight.pt"

        if not weight_path.exists():
            print(
                f"[INFO] Downloading {model_type} model..."
            )

            download_file(
                download_link,
                weight_path
            )

        model_paths[model_type] = str(weight_path)

    return model_paths
