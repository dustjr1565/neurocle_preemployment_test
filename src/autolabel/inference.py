import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

from autolabel.data import change_db_to_base_format, change_mask_to_points
from autolabel.db import insert_image, insert_annotation, create_database, LabelStatus
from autolabel.quality import run_quality_check

def inference_model(
        image_dir: str,
        output_dir: str,
        det_model_path: str, 
        sam_model_path: str, 
        vis: bool
    ):
    detector = YOLO(det_model_path)
    sam = SAM(sam_model_path)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir) 
    output_dir.mkdir(parents=True)
    db_path = output_dir / "autolabel.db"
    create_database(str(db_path))

    if vis:
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True)

    print(f"추론할 이미지 개수: {len(list(image_dir.glob('*.jpg')))}")
    for image_path in list(image_dir.glob("*.jpg")):
        image_path = image_dir / image_path.name
        image = cv2.imread(str(image_path))
        height, width, _ = image.shape
        det_results = detector(image, verbose=False)
        det_result = det_results[0]

        image_id = insert_image(
            file_name=str(image_path.name),
            width=width,
            height=height,
            db_path=db_path,
            status=LabelStatus.COMPLETED
        )

        if det_result.boxes is None or len(det_result.boxes) == 0:
            continue

        class_ids = det_result.boxes.cls.cpu().numpy().astype(int)
        confidences = det_result.boxes.conf.cpu().numpy()
        boxes = det_result.boxes.xyxy.cpu().numpy()
        sam_results = sam(
            image,
            bboxes=boxes
        )

        result = sam_results[0]
        masks = result.masks.data.cpu().numpy()

        qc_results = run_quality_check(image, confidences, boxes, masks)

        for class_id, conf, bbox, mask, qc_result in zip(class_ids, confidences, boxes, masks, qc_results):
            x1, y1, x2, y2 = map(int, bbox)
            w, h = int(x2 - x1), int(y2 - y1)
            class_name = det_result.names[int(class_id)]
            insert_annotation(
                image_id=image_id,
                class_id=int(class_id),
                class_name=class_name,
                bbox=f"[{x1},{y1},{w},{h}]",
                segment=change_mask_to_points(mask),
                confidence=float(conf),
                auto_labeled=True,
                need_review=qc_result,
                db_path=db_path
            )

            if qc_result:
                print(f"QQQQCCCC: {image_path.name}, class_name: {class_name}")
            

        if vis:
            if result.masks is not None:
                for mask in masks:
                    points = change_mask_to_points(mask)
                    pts = np.array(points, dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    mask = mask.astype(bool)

                    overlay = image.copy()

                    cv2.fillPoly(
                        overlay,
                        [pts],
                        (0, 0, 255)
                    )

                    image = cv2.addWeighted(
                        image,
                        0.3,
                        overlay,
                        0.7,
                        0
                    )

                    cv2.polylines(
                        image,
                        [pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2
                    )

            # bbox + class명
            for class_id, conf, bbox in zip(class_ids, confidences, boxes):
                x1, y1, x2, y2 = map(int, bbox)
                class_name = det_result.names[int(class_id)]
                label = f"{class_name} {conf:.2f}"
                
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            vis_path = vis_dir / image_path.name
            cv2.imwrite(str(vis_path), image)

    json_path = output_dir / "autolabel.json"
    change_db_to_base_format(db_path, json_path)