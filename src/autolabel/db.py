import sqlite3
import json
from enum import IntEnum

class LabelStatus(IntEnum):
    UNLABELED = 0
    PENDING = 1
    COMPLETED = 2


def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    cursor = conn.cursor()

    # Image 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Image (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            width INT NOT NULL,
            height INT NOT NULL,
            status INT NOT NULL
        )
    """)

    # annotation 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotation (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT,
            bbox TEXT,
            segment TEXT,
            confidence REAL,
            auto_labeled BOOL,
            need_review BOOL,

            FOREIGN KEY (image_id)
                REFERENCES Image(image_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def insert_image(file_name: str, width: int, height: int, db_path: str, status=0):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO Image (
            file_name,
            width,
            height,
            status
        )
        VALUES (?, ?, ?, ?)
    """, (
        file_name,
        width,
        height,
        status
    ))

    image_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return image_id

def insert_annotation(
    image_id,
    class_id,
    class_name,
    bbox,
    segment,
    confidence,
    db_path,
    auto_labeled=True,
    need_review=False
):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO annotation (
            image_id,
            class_id,
            class_name,
            bbox,
            segment,
            confidence,
            auto_labeled,
            need_review
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_id,
        class_id,
        class_name,
        json.dumps(bbox),
        json.dumps(segment),
        confidence,
        auto_labeled,
        need_review
    ))

    conn.commit()
    conn.close()


def get_image(db_path, image_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM Image
            WHERE image_id = ?
        """, (image_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "width": row["width"],
            "height": row["height"],
            "status": row["status"],
        }

    finally:
        conn.close()


def get_image_count(db_path):
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM Image
        """)

        return cursor.fetchone()[0]

    finally:
        conn.close()

def get_annotations(db_path, image_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM annotation
            WHERE image_id = ?
        """, (image_id,))

        rows = cursor.fetchall()

        annotations = []

        for row in rows:
            annotations.append({
                "annotation_id": row["annotation_id"],
                "image_id": row["image_id"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "bbox": json.loads(row["bbox"]),
                "segment": json.loads(row["segment"]),
                "confidence": row["confidence"],
                "auto_labeled": bool(row["auto_labeled"]),
                "need_review": bool(row["need_review"]),
            })

        return annotations

    finally:
        conn.close()