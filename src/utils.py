"""
src/utils.py
============
Shared helper functions: on-screen UI drawing for the recognition HUD,
face bounding boxes, and status panels. Keeping these outside the main
control loop keeps main.py focused and readable.
"""

from typing import Tuple

import cv2
import numpy as np

from config import settings

# A face bounding box in (top, right, bottom, left) order, matching the
# convention used throughout the face_recognition library.
FaceBox = Tuple[int, int, int, int]


def draw_rounded_panel(frame: np.ndarray, top_left: Tuple[int, int],
                        bottom_right: Tuple[int, int],
                        color: Tuple[int, int, int],
                        alpha: float = 0.6) -> np.ndarray:
    """Draw a semi-transparent filled rectangle used as a UI backdrop."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_face_box(frame: np.ndarray, box: FaceBox, name: str,
                   confidence: float, status_label: str,
                   color: Tuple[int, int, int]) -> np.ndarray:
    """
    Draws a bounding box around a detected face, with a name/confidence
    label above it and an attendance-status label below it.
    """
    top, right, bottom, left = box

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    if name == "Unknown":
        name_label = "Unknown"
    else:
        name_label = f"{name.replace('_', ' ')} ({confidence * 100:.0f}%)"

    label_bg_top = max(0, top - 32)
    cv2.rectangle(frame, (left, label_bg_top), (right, top), color, cv2.FILLED)
    cv2.putText(frame, name_label, (left + 6, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(frame, status_label, (left, bottom + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


def draw_status_panel(frame: np.ndarray, fps: float, webcam_ok: bool,
                       faces_detected: int, best_confidence: float,
                       students_registered: int, attendance_today: int) -> np.ndarray:
    """
    Renders the heads-up display: FPS counter, webcam status, number of
    faces currently in frame, best match confidence, registered student
    count, and today's attendance count.
    """
    panel_height = 180
    frame = draw_rounded_panel(
        frame, (0, 0), (340, panel_height), settings.COLOR_PANEL_BG, alpha=0.55
    )

    fps_color = settings.COLOR_SUCCESS if fps >= 10 else settings.COLOR_WARNING
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    status_text = "Webcam: ONLINE" if webcam_ok else "Webcam: OFFLINE"
    status_color = settings.COLOR_SUCCESS if webcam_ok else settings.COLOR_ERROR
    cv2.putText(frame, status_text, (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    cv2.putText(frame, f"Faces in Frame: {faces_detected}", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, settings.COLOR_TEXT, 1)

    cv2.putText(frame, f"Best Match Confidence: {best_confidence * 100:.0f}%",
                (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, settings.COLOR_TEXT, 1)

    cv2.putText(frame, f"Registered Students: {students_registered}", (12, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, settings.COLOR_PRIMARY, 1)

    cv2.putText(frame, f"Marked Present Today: {attendance_today}", (12, 152),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, settings.COLOR_SUCCESS, 2)

    return frame


def draw_control_hints(frame: np.ndarray) -> np.ndarray:
    """Renders the keyboard shortcut hints in the bottom-left corner."""
    h, _ = frame.shape[:2]
    exit_text = f"Press '{settings.EXIT_KEY.upper()}' or ESC to exit"
    register_text = f"Press '{settings.REGISTER_KEY.upper()}' to register a new student"

    cv2.putText(frame, register_text, (12, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings.COLOR_TEXT, 1)
    cv2.putText(frame, exit_text, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings.COLOR_TEXT, 1)
    return frame
