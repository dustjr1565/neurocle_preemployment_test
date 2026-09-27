import argparse
import random
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

from autolabel.db import get_image_count, get_image, get_annotations

CLASS_NAMES = [
    "MouthWash",
    "HairRoll",
    "VaselinJar",
    "PillJar",
    "HandCream",
    "Tissue",
    "MouthWash Box",
    "Phone",
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

KST = timezone(timedelta(hours=9))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="./assignment_data",
        help="입력 파일 경로"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="출력 파일 경로"
    )

    parser.add_argument(
        "--vis",
        action="store_true",
        help="visualization 활성화"
    )

    parser.add_argument(
        "--yolo",
        action="store_true",
        help="yolo 포맷 변경 활성화"
    )

    return parser.parse_args()


def change_mask_to_points(mask: np.ndarray) -> list[list[int]]:
    """
    bool mask -> polygon point list

    Args:
        mask: (H, W) bool numpy array

    Returns:
        [[x1, y1], [x2, y2], ...]
    """

    # bool -> uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # 외곽선 추출
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    # 가장 큰 영역의 contour 사용
    contour = max(contours, key=cv2.contourArea)

    # (N, 1, 2) -> (N, 2) -> list
    points = contour.reshape(-1, 2).tolist()

    return points

def get_base_format_infos(label_path: str) -> list[dict]:
    infos = []
    with open(label_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    count = 0
    for data in datas["data"]:
        if not data.get("regionLabel", {}):
            break

        info = {
            "fileName": data["fileName"],
            "objects": [],
            "width": data["width"],
            "height": data["height"]
        }

        for obj_data in data["regionLabel"]:
            obj_info = {
                "className": obj_data["className"],
                "x": obj_data["x"],
                "y": obj_data["y"],
                "w": obj_data["width"],
                "h": obj_data["height"],
            }
            info["objects"].append(obj_info)
        count += 1

        infos.append(info)

    print(f"labeled data count: {count}")
    return infos

def visualize_data(root_dir: str, output_dir: str):
    image_dir = Path(root_dir) / "images"
    label_path = Path(root_dir) / "label.json"
    output_dir = Path(output_dir) / "vis"

    if output_dir.exists():
        print("[Error] 이미 출력파일이 존재합니다.")
    else:
        output_dir.mkdir(parents=True)

    infos = get_base_format_infos(str(label_path))
    for info in infos:
        image_path = image_dir / info["fileName"]
        image = cv2.imread(str(image_path))
        for obj_info in info["objects"]:
            cv2.rectangle(
                image,
                (obj_info["x"], obj_info["y"]),
                (obj_info["x"]+obj_info["w"], obj_info["y"]+obj_info["h"]),
                (0,0,255),
                2
            )
            cv2.putText(
                image,
                obj_info["className"],
                (obj_info["x"], obj_info["y"]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        output_path = output_dir / info["fileName"]
        cv2.imwrite(str(output_path), image)
    return image


def change_base_to_yolo_format(root_dir: str, output_dir: str):
    image_dir = Path(root_dir) / "images"
    label_path = Path(root_dir) / "label.json"
    output_dir = Path(output_dir) / "yolo"

    if output_dir.exists():
        print("[Error] 이미 출력파일이 존재합니다.")
        return

    output_dir.mkdir(parents=True)

    train_image_dir = output_dir / "images" / "train"
    val_image_dir = output_dir / "images" / "val"

    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"

    for path in [
        train_image_dir,
        val_image_dir,
        train_label_dir,
        val_label_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    infos = get_base_format_infos(str(label_path))

    # 8:2 분할
    random.seed(42)
    random.shuffle(infos)

    train_count = int(len(infos) * 4 / 5)

    train_infos = infos[:train_count]
    val_infos = infos[train_count:]

    for info, image_output_dir, label_output_dir in [
        *[(info, train_image_dir, train_label_dir) for info in train_infos],
        *[(info, val_image_dir, val_label_dir) for info in val_infos],
    ]:
        image_path = image_dir / info["fileName"]

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[Error] 이미지를 읽을 수 없습니다: {image_path}")
            continue

        image_h, image_w = image.shape[:2]

        # 이미지 복사
        shutil.copy2(
            image_path,
            image_output_dir / info["fileName"]
        )

        # YOLO label 생성
        label_file = label_output_dir / f"{Path(info['fileName']).stem}.txt"

        with open(label_file, "w", encoding="utf-8") as f:
            for obj_info in info["objects"]:
                class_name = obj_info["className"]

                if class_name not in CLASS_TO_ID:
                    print(f"[Warning] 알 수 없는 클래스: {class_name}")
                    continue

                class_id = CLASS_TO_ID[class_name]

                x = obj_info["x"]
                y = obj_info["y"]
                w = obj_info["w"]
                h = obj_info["h"]

                center_x = (x + w / 2) / image_w
                center_y = (y + h / 2) / image_h
                norm_w = w / image_w
                norm_h = h / image_h

                f.write(
                    f"{class_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{norm_w:.6f} "
                    f"{norm_h:.6f}\n"
                )

    # data.yaml
    yaml_path = output_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")

        for idx, class_name in enumerate(CLASS_NAMES):
            f.write(f"  {idx}: {class_name}\n")

    print(f"[Done] Total : {len(infos)}")
    print(f"[Done] Train : {len(train_infos)}")
    print(f"[Done] Val   : {len(val_infos)}")
    print(f"[Done] Output: {output_dir}")

def change_db_to_base_format(db_path: str, output_path: str):
    image_num = get_image_count(db_path)

    label = {
        "label_type": "obd",
        "source": "labelset",
        "version": "5.0.2.32",
        "classes": [
            {
                "name": "MouthWash",
                "color": "rgba(167, 238, 62, 1)"
            },
            {
                "name": "HairRoll",
                "color": "rgba(255, 180, 12, 1)"
            },
            {
                "name": "VaselinJar",
                "color": "rgba(251, 92, 73, 1)"
            },
            {
                "name": "PillJar",
                "color": "rgba(60, 109, 240, 1)"
            },
            {
                "name": "HandCream",
                "color": "rgba(52, 188, 110, 1)"
            },
            {
                "name": "Tissue",
                "color": "rgba(86, 204, 242, 1)"
            },
            {
                "name": "MouthWash Box",
                "color": "rgba(248, 126, 172, 1)"
            },
            {
                "name": "Phone",
                "color": "rgba(167, 238, 62, 1)"
            }
        ],
        "data": [],
        "time": datetime.now(KST).isoformat()
    }

    for image_id in range(1, 1+image_num):
        image = get_image(db_path,image_id)
        annotations = get_annotations(
            db_path,
            image_id=image_id
        )
        data = {
            "fileName": image["file_name"],
            "set": "",
            "classLabel": "",
            "regionLabel": [],
            "retestset": 0,
            "rotation_angle": 0.0,
            "width": image["width"],
            "height": image["height"],
        }

        for ann in annotations:
            x1, y1, w, h = list(map(int, eval(ann["bbox"])))
            data["regionLabel"].append(
                {
                    "className": ann["class_name"],
                    "x": x1,
                    "y": y1,
                    "type": "Rect",
                    "width": w,
                    "height": h
                }
            )

        label["data"].append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            label,
            f,
            ensure_ascii=False,
            indent=4
        )
        

def main():
    args = parse_args()

    print("input:", args.input)
    if not Path(args.input).exists():
        print(f"[Error] 입력이미지 경로가 존재하지 않음")

    if args.vis:
        visualize_data(args.input, args.output)

    if args.yolo:
        change_base_to_yolo_format(args.input, args.output)
    


if __name__ == "__main__":
    main()