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

def inference(args):
    model_dir = Path(args.model_dir)

    # --------------------------------------------------
    # 모델이 존재하지 않는 경우
    # --------------------------------------------------
    if not model_dir.exists():
        print(
            "[INFO] 모델이 존재하지않습니다. "
            "최근 모델을 다운로드 받습니다."
        )

        det_model_path, sam_model_path = download_models(
            model_dir
        )

    # --------------------------------------------------
    # 모델이 이미 존재하는 경우
    # --------------------------------------------------
    else:
        print("[INFO] 기존 모델을 사용합니다.")

        # 기존 info.yaml을 읽어서 모델 경로 확인
        info_path = model_dir / "info.yaml"

        if not info_path.exists():
            print(
                "[INFO] info.yaml이 존재하지않습니다. "
                "최근 모델 정보를 다운로드합니다."
            )

            det_model_path, sam_model_path = download_models(
                model_dir
            )

        else:
            with open(info_path, "r", encoding="utf-8") as f:
                info = yaml.safe_load(f)

            models = info["model"]

            det_model_path = str(
                model_dir / models["detection"]["name"]
            )

            sam_model_path = str(
                model_dir / models["sam"]["name"]
            )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    inference_model(
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        det_model_path=det_model_path,
        sam_model_path=sam_model_path,
        vis=args.vis,
    )