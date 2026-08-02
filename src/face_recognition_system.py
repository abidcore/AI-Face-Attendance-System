"""
src/face_recognition_system.py
================================
Owns the "known faces" database: building face encodings from the student
image dataset, caching them to disk for fast startup, and matching newly
detected faces against that database in real time.

Key design decisions
---------------------
- Multiple images per student are encoded and *averaged* into a single
  representative encoding. This is meaningfully more robust than using a
  single reference image, since it smooths out lighting, angle, and
  expression variance across the registration samples.
- Encodings are cached to disk (`database/encodings.pkl`) so the
  (relatively expensive) encoding process does not need to re-run on
  every application startup. The cache is automatically invalidated and
  rebuilt whenever the set of registered student folders changes.
- Faces that don't match any known student within `RECOGNITION_TOLERANCE`
  are explicitly classified as "Unknown" rather than forced into the
  closest (possibly incorrect) match.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

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


@dataclass
class RecognitionResult:
    """Outcome of matching a single detected face against known students."""
    name: str
    confidence: float   # 0.0 - 1.0, derived from (1 - face_distance)
    is_known: bool


class FaceRecognitionSystem:
    """Builds, caches, and queries the known-student face encoding database."""

    def __init__(self) -> None:
        self.known_encodings: Dict[str, np.ndarray] = {}
        self._load_or_build_encodings()

    # ------------------------------------------------------------------
    # Encoding database management
    # ------------------------------------------------------------------
    def _registered_student_names(self) -> List[str]:
        """Returns the sorted list of student subfolder names on disk."""
        if not os.path.isdir(settings.DATASET_DIR):
            os.makedirs(settings.DATASET_DIR, exist_ok=True)
            return []
        return sorted(
            entry for entry in os.listdir(settings.DATASET_DIR)
            if os.path.isdir(os.path.join(settings.DATASET_DIR, entry))
            and not entry.startswith(".")
        )

    def _load_or_build_encodings(self) -> None:
        """
        Attempts to load a valid cached encoding database. The cache is
        considered valid only if the set of student names it was built
        from exactly matches the student folders currently on disk;
        otherwise the encodings are rebuilt from scratch.
        """
        current_students = self._registered_student_names()

        if os.path.exists(settings.ENCODINGS_CACHE_PATH):
            try:
                with open(settings.ENCODINGS_CACHE_PATH, "rb") as cache_file:
                    cached = pickle.load(cache_file)

                if cached.get("student_names") == current_students:
                    self.known_encodings = cached["encodings"]
                    print(
                        f"[INFO] Loaded cached encodings for "
                        f"{len(self.known_encodings)} student(s)."
                    )
                    return
            except (pickle.PickleError, EOFError, KeyError, OSError) as error:
                print(f"[WARNING] Encoding cache could not be read ({error}). "
                      "Rebuilding from the image dataset.")

        self.build_encodings()
        self._save_cache()

    def build_encodings(self) -> None:
        """
        Scans the student image dataset and builds one averaged face
        encoding per registered student. Images that contain no
        detectable face, or that fail to load, are skipped with a
        warning rather than aborting the entire process.
        """
        self.known_encodings = {}
        student_names = self._registered_student_names()

        for student_name in student_names:
            student_dir = os.path.join(settings.DATASET_DIR, student_name)
            sample_encodings: List[np.ndarray] = []

            for filename in sorted(os.listdir(student_dir)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(student_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        sample_encodings.append(face_encodings[0])
                    else:
                        print(f"[WARNING] No face found in '{image_path}'. Skipped.")
                except Exception as error:  # noqa: BLE001
                    print(f"[WARNING] Could not process '{image_path}': {error}")

            if sample_encodings:
                # Averaging multiple samples produces a more robust
                # representative encoding than any single image.
                self.known_encodings[student_name] = np.mean(sample_encodings, axis=0)
            else:
                print(f"[WARNING] No usable face images found for '{student_name}'; "
                      "this student will not be recognizable until re-registered.")

        print(f"[INFO] Built encodings for {len(self.known_encodings)} student(s).")

    def _save_cache(self) -> None:
        """Persists the current encoding database to disk for fast reload."""
        os.makedirs(os.path.dirname(settings.ENCODINGS_CACHE_PATH), exist_ok=True)
        payload = {
            "student_names": sorted(self.known_encodings.keys()),
            "encodings": self.known_encodings,
        }
        with open(settings.ENCODINGS_CACHE_PATH, "wb") as cache_file:
            pickle.dump(payload, cache_file)

    def refresh(self) -> None:
        """Rebuilds and re-caches the encoding database. Call after registration."""
        self.build_encodings()
        self._save_cache()

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize(self, rgb_small_frame: np.ndarray,
                   small_locations: List[tuple]) -> List[RecognitionResult]:
        """
        Computes face encodings for every detected face in the given
        (downscaled) RGB frame and matches each against the known-student
        database. Returns one RecognitionResult per detected face, in the
        same order as `small_locations`.
        """
        face_encodings = face_recognition.face_encodings(rgb_small_frame, small_locations)
        results: List[RecognitionResult] = []

        known_names = list(self.known_encodings.keys())
        known_values = list(self.known_encodings.values())

        for encoding in face_encodings:
            if not known_values:
                results.append(RecognitionResult("Unknown", 0.0, False))
                continue

            distances = face_recognition.face_distance(known_values, encoding)
            best_index = int(np.argmin(distances))
            best_distance = float(distances[best_index])
            confidence = max(0.0, 1.0 - best_distance)

            if best_distance <= settings.RECOGNITION_TOLERANCE:
                results.append(RecognitionResult(known_names[best_index], confidence, True))
            else:
                results.append(RecognitionResult("Unknown", confidence, False))

        return results
