import cv2
from pathlib import Path
from ultralytics import YOLO, SAM
from data import get_base_format_infos

sam = SAM("sam2_b.pt")

infos = get_base_format_infos("./assignment_data/label.json")
image_dir = Path("assignment_data/images")
output_dir = Path("outputs/sam_gt")
output_dir.mkdir(parents=True)

for info in infos:
    image_path = image_dir / info["fileName"]
    image = cv2.imread(str(image_path))

    boxes = []
    for obj_info in info["objects"]:
        x = obj_info["x"]
        y = obj_info["y"]
        w = obj_info["w"]
        h = obj_info["h"]

        boxes.append([
            float(x),
            float(y),
            float(x + w),
            float(y + h),
        ])


    if len(boxes) == 0:
        continue

    sam_results = sam(
        image,
        bboxes=boxes
    )

    result = sam_results[0]

    # mask 시각화
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

        for mask in masks:
            mask = mask.astype(bool)

            # mask 크기가 원본과 다를 경우
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # 빨간색 mask
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
    for obj_info, bbox in zip(info["objects"], boxes):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            obj_info["className"],
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    output_path = output_dir / info["fileName"]
    cv2.imwrite(str(output_path), image)