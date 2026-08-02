"""
src/face_detector.py
======================
Wraps the `face_recognition` library's face-location detection behind a
clean, reusable interface, with a built-in performance optimization:
detection runs on a downscaled copy of each frame (since dlib-based
detection is CPU-intensive), and resulting coordinates are scaled back up
to the original frame's resolution for accurate on-screen drawing.
"""

from typing import List, Tuple

import cv2
import numpy as np

from config import settings

try:
    import face_recognition
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "The 'face_recognition' package (and its dlib dependency) could "
        "not be imported. Install it with 'pip install face_recognition'. "
        "On Windows you may need CMake and Visual C++ Build Tools "
        "installed first; on Linux, install 'cmake' and 'build-essential' "
        "via your package manager before installing dlib."
    ) from import_error


# A face bounding box in (top, right, bottom, left) order, matching the
# convention used throughout the face_recognition library.
FaceBox = Tuple[int, int, int, int]


class FaceDetector:
    """Detects face locations in a video frame using dlib/face_recognition."""

    def __init__(self) -> None:
        self.model = settings.FACE_DETECTION_MODEL

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[FaceBox], List[FaceBox]]:
        """
        Detects all faces in a BGR frame.

        Returns a 3-tuple:
            rgb_small_frame  - the downscaled RGB frame actually used for
                                detection (required by face_recognition
                                for subsequent encoding calls on the same
                                coordinate space)
            small_locations  - face boxes in the downscaled frame's
                                coordinate space (for encoding)
            original_locations - the same boxes scaled back up to the
                                  original frame's resolution (for drawing)
        """
        small_frame = cv2.resize(
            frame, (0, 0),
            fx=settings.FRAME_RESIZE_SCALE,
            fy=settings.FRAME_RESIZE_SCALE,
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        small_locations = face_recognition.face_locations(
            rgb_small_frame, model=self.model
        )

        scale = 1.0 / settings.FRAME_RESIZE_SCALE
        original_locations: List[FaceBox] = [
            (
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale),
            )
            for (top, right, bottom, left) in small_locations
        ]

        return rgb_small_frame, small_locations, original_locations
