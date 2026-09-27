import argparse
from pathlib import Path

from autolabel.inference import inference_model
from autolabel.file import download_file, download_models

INFO_YAML_URL = (
    "https://raw.githubusercontent.com/"
    "dustjr1565/neurocle_preemployment_test/main/model/info.yaml"
)

def inference(args):
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory : {args.model_dir}")
    print(f"Visualization   : {args.vis}")

    model_dir = Path(args.model_dir)
    model_info_path = dst_path = model_dir / "info.yaml"
    if not model_info_path.exists():
        print("[INFO] 모델 정보가 존재하지않습니다. 최근 모델 파일 다운로드 받습니다.")
        print("[INFO] 모델 정보 파일 다운로드 받는 중...")
        download_file(INFO_YAML_URL, dst_path)
    model_paths = download_models(model_dir)

    inference_model(
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        det_model_path=str(model_paths["detection"]),
        sam_model_path="model/base_sam_v1/sam2_b.pt",
        vis=args.vis,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="autolabel",
        description="오토 라벨링 CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # --------------------------------------------------
    # inference
    # --------------------------------------------------
    inference_parser = subparsers.add_parser(
        "inference",
        help="이미지 추론 및 자동 라벨링"
    )

    inference_parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="추론할 이미지들이 있는 디렉터리 경로"
    )

    inference_parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inference",
        help="추론 결과를 저장할 디렉터리 경로"
    )

    inference_parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="모델 디렉터리 경로. 존재하지 않을 경우 다운로드"
    )

    inference_parser.add_argument(
        "--vis",
        action="store_true",
        help="추론 결과를 시각화하여 저장"
    )

    inference_parser.set_defaults(func=inference)

    # --------------------------------------------------
    # command 실행
    # --------------------------------------------------
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
