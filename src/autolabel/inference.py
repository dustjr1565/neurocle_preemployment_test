import cv2
from pathlib import Path
from ultralytics import YOLO, SAM


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

    if vis:
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(parents=True)

    for image_path in list(image_dir.glob("*.jpg")):
        image_path = image_dir / image_path.name
        image = cv2.imread(str(image_path))

        det_results = detector(image, verbose=False)
        det_result = det_results[0]

        # Detection 결과가 없는 경우
        if det_result.boxes is None or len(det_result.boxes) == 0:
            continue

        # xyxy: (N, 4)
        class_ids = det_result.boxes.cls.cpu().numpy().astype(int)
        confidences = det_result.boxes.conf.cpu().numpy()
        boxes = det_result.boxes.xyxy.cpu().numpy()
        sam_results = sam(
            image,
            bboxes=boxes
        )

        result = sam_results[0]

        if vis:
            # mask 시각화
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

                for mask in masks:
                    mask = mask.astype(bool)

                    overlay = image.copy()
                    overlay[mask] = (0, 0, 255)

                    image = cv2.addWeighted(
                        image,
                        0.3,
                        overlay,
                        0.7,
                        0
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