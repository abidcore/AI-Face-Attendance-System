"""
config/settings.py
===================
Centralized configuration for the AI Face Attendance Management System.

Keeping every tunable parameter in a single module makes the application
easy to calibrate for different cameras, datasets, and performance targets
without touching the core logic in `src/`.
"""

import os

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATASET_DIR = os.path.join(DATABASE_DIR, "students")
ATTENDANCE_CSV_PATH = os.path.join(DATABASE_DIR, "attendance.csv")
ENCODINGS_CACHE_PATH = os.path.join(DATABASE_DIR, "encodings.pkl")

# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                 # Default webcam device index
FRAME_WIDTH = 640                 # Capture frame width (pixels)
FRAME_HEIGHT = 480                # Capture frame height (pixels)
FLIP_CAMERA = True                # Mirror the webcam feed for a natural UX

# ---------------------------------------------------------------------------
# Face Detection & Recognition Configuration
# ---------------------------------------------------------------------------
# "hog" is fast and CPU-friendly (recommended for laptops/real-time use).
# "cnn" is more accurate but significantly slower without a CUDA-enabled
# GPU. Switch to "cnn" only if real-time performance is not required.
FACE_DETECTION_MODEL = "hog"

# Frames are downscaled before detection/encoding to keep the pipeline
# real-time, since face_recognition (dlib) is CPU-intensive at full
# resolution. Detected coordinates are scaled back up for display.
FRAME_RESIZE_SCALE = 0.25

# Maximum face-distance (0.0 = identical, higher = less similar) for a
# match to be accepted as a known person. Lower is stricter. The
# face_recognition library's own default is 0.6; a slightly stricter
# value reduces false-positive matches between similar-looking people.
RECOGNITION_TOLERANCE = 0.50

# Detection + recognition is only run every N frames; results are reused
# on the frames in between. This is the single biggest performance lever
# for keeping the UI responsive, since dlib-based encoding is expensive.
PROCESS_EVERY_N_FRAMES = 3

# ---------------------------------------------------------------------------
# Registration Configuration
# ---------------------------------------------------------------------------
IMAGES_PER_REGISTRATION = 5        # Number of face samples captured per student
REGISTRATION_CAPTURE_DELAY = 1.0   # Seconds between automatic captures

# ---------------------------------------------------------------------------
# Attendance Configuration
# ---------------------------------------------------------------------------
ATTENDANCE_CSV_HEADERS = ["Name", "Date", "Time", "Status"]

# ---------------------------------------------------------------------------
# UI / Overlay Configuration
# ---------------------------------------------------------------------------
SHOW_FPS = True
SHOW_STATUS_PANEL = True

WINDOW_NAME = "AI Face Attendance Management System"

# Colors are defined in BGR (OpenCV convention)
COLOR_PRIMARY = (255, 150, 0)
COLOR_SUCCESS = (0, 200, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL_BG = (30, 30, 30)

# ---------------------------------------------------------------------------
# Application Behaviour
# ---------------------------------------------------------------------------
EXIT_KEY = "q"                     # Keyboard shortcut to quit safely (ESC also works)
REGISTER_KEY = "r"                 # Keyboard shortcut to register a new student
