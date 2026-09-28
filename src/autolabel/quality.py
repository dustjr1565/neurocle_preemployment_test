import cv2
import numpy as np


def run_quality_check(
    image: np.ndarray,
    confs: np.ndarray,
    bboxs: np.ndarray,
    masks: list[np.ndarray],
) -> list[bool]:
    """
    4가지 조건으로 데이터를 확인 후
    각 bbox별 검수 필요 여부를 반환한다.

    QC 조건
    1. Confidence가 낮은 경우
    2. BBox 크기가 너무 작은 경우
    3. BBox 끼리 겹침 또는 포함관계에 있는 경우
    4. BBox 내부의 노란색 영역이 아니면서, mask에서 누락된 경우

    Returns:
        list[bool]:
            각 bbox에 대한 검수 필요 여부
    """

    # =========================
    # Threshold
    # =========================

    CONF_THRESHOLD = 0.95

    MIN_BBOX_WIDTH = 16
    MIN_BBOX_HEIGHT = 16

    LOWER_BACKGROUND = np.array([0, 5, 5])
    UPPER_BACKGROUND = np.array([45, 255, 255])
    MISSING_AREA_RATIO_THRESHOLD = 0.2  # 20%

    num_objects = len(bboxs)

    results = [False] * num_objects

    # =========================
    # 1. Confidence Check
    # =========================

    for i, conf in enumerate(confs):

        if float(conf) < CONF_THRESHOLD:
            results[i] = True

    # =========================
    # 2. BBox Size Check
    # =========================

    for i, bbox in enumerate(bboxs):

        x1, y1, x2, y2 = map(int, bbox)

        width = x2 - x1
        height = y2 - y1

        if width < MIN_BBOX_WIDTH or height < MIN_BBOX_HEIGHT:
            results[i] = True

    # =========================================================
    # 3. BBox Overlap / Inclusion Check
    # =========================================================

    def is_overlap(box1, box2):
        """
        두 bbox가 실제로 겹치는지 확인.
        IoU threshold를 사용하지 않는다.
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        return x1 < x2 and y1 < y2

    for i in range(num_objects):

        for j in range(i + 1, num_objects):

            if is_overlap(bboxs[i], bboxs[j]):

                results[i] = True
                results[j] = True

    # =========================================================
    # 4. Mask Missing Area Check
    # =========================================================
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(
        hsv,
        LOWER_BACKGROUND,
        UPPER_BACKGROUND
    )

    for i, (bbox, mask) in enumerate(zip(bboxs, masks)):

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            results[i] = True
            continue

        # BBox 내부
        roi_yellow = (
            yellow_mask[y1:y2, x1:x2] > 0
        )

        roi_mask = (
            mask[y1:y2, x1:x2].astype(bool)
        )

        # 노란색이 아니면서 mask에도 없는 영역
        missing_area = (
            ~roi_yellow
            & ~roi_mask
        )

        missing_pixel_count = np.count_nonzero(
            missing_area
        )

        mask_pixel_count = np.count_nonzero(
            roi_mask
        )


        if mask_pixel_count > 0:
            missing_ratio = (
                missing_pixel_count / mask_pixel_count
            )
        else:
            missing_ratio = 1

        if missing_ratio >= MISSING_AREA_RATIO_THRESHOLD:
            results[i] = True
            
    return results