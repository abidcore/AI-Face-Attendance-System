"""
src/dataset_manager.py
========================
Owns all file-system operations related to the student image dataset:
creating per-student folders, saving captured face images (via Pillow),
sanitizing student names into safe folder names, and listing already
registered students.

This module is intentionally limited to file I/O only - the interactive
webcam capture loop used during registration lives in `main.py`, since it
needs to coordinate directly with the live OpenCV preview window.
"""

import os
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from config import settings


class DatasetManager:
    """Manages the on-disk student face-image dataset."""

    def __init__(self) -> None:
        os.makedirs(settings.DATASET_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Name handling
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_name(raw_name: str) -> str:
        """
        Converts a free-form student name into a safe, filesystem-friendly
        folder name (spaces become underscores; unsupported characters are
        removed). Raises ValueError if nothing usable remains.
        """
        cleaned = raw_name.strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)

        if not cleaned:
            raise ValueError(
                "The provided name contains no valid characters after "
                "sanitization. Please use letters, numbers, spaces, "
                "hyphens, or underscores."
            )

        return cleaned

    # ------------------------------------------------------------------
    # Folder / file management
    # ------------------------------------------------------------------
    def get_student_dir(self, student_name: str) -> str:
        """Returns (and creates, if necessary) a student's image folder."""
        student_dir = os.path.join(settings.DATASET_DIR, student_name)
        os.makedirs(student_dir, exist_ok=True)
        return student_dir

    def student_exists(self, student_name: str) -> bool:
        """Returns True if a folder already exists for this student."""
        return os.path.isdir(os.path.join(settings.DATASET_DIR, student_name))

    def save_face_image(self, student_name: str, frame_bgr: np.ndarray,
                         sample_index: int) -> str:
        """
        Saves a single captured frame as a JPEG image in the student's
        folder, using Pillow for the actual image encoding/writing.

        Returns the full path to the saved image file.
        """
        student_dir = self.get_student_dir(student_name)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_name}_{timestamp}_{sample_index}.jpg"
        filepath = os.path.join(student_dir, filename)

        pil_image.save(filepath, format="JPEG", quality=95)
        return filepath

    def list_registered_students(self) -> list:
        """Returns a sorted list of all currently registered student names."""
        if not os.path.isdir(settings.DATASET_DIR):
            return []
        return sorted(
            entry for entry in os.listdir(settings.DATASET_DIR)
            if os.path.isdir(os.path.join(settings.DATASET_DIR, entry))
            and not entry.startswith(".")
        )

    def count_images_for_student(self, student_name: str) -> int:
        """Returns how many valid image files exist for a given student."""
        student_dir = os.path.join(settings.DATASET_DIR, student_name)
        if not os.path.isdir(student_dir):
            return 0
        return len([
            f for f in os.listdir(student_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
