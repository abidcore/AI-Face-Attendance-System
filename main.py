"""
main.py
=======
Entry point for the AI Face Attendance Management System.

Runs a live webcam feed that continuously detects and recognizes faces,
automatically marking attendance (once per student per day) in a CSV
file. Pressing 'R' at any time pauses recognition and walks through an
interactive new-student registration flow; pressing 'Q' or ESC exits the
application safely.

Run this file directly to start the application:

    python main.py
"""

import sys
import time

import cv2

from config import settings
from src.face_detector import FaceDetector
from src.face_recognition_system import FaceRecognitionSystem
from src.attendance_manager import AttendanceManager
from src.dataset_manager import DatasetManager
from src.fps import FPSCounter
from src.utils import draw_status_panel, draw_control_hints, draw_face_box


class FaceAttendanceApp:
    """Top-level application object: owns the main capture/processing loop."""

    def __init__(self) -> None:
        self.capture = None
        self.webcam_ok = False

        self.face_detector = FaceDetector()
        self.recognition_system = FaceRecognitionSystem()
        self.attendance_manager = AttendanceManager()
        self.dataset_manager = DatasetManager()
        self.fps_counter = FPSCounter()

        self._frame_counter = 0
        self._cached_results = []  # list of (face_box, RecognitionResult)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _open_camera(self) -> None:
        """Open the webcam and configure its resolution, with error handling."""
        self.capture = cv2.VideoCapture(settings.CAMERA_INDEX)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {settings.CAMERA_INDEX}. "
                "Check that a camera is connected, that no other "
                "application is using it, and that OS camera permissions "
                "are granted."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        self.webcam_ok = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Starts the application and runs until the user exits."""
        try:
            self._open_camera()
        except RuntimeError as error:
            print(f"[FATAL] {error}")
            sys.exit(1)

        print("=" * 60)
        print(" AI Face Attendance Management System - Started")
        print("=" * 60)
        print(f" Registered students: {len(self.recognition_system.known_encodings)}")
        print(f" Already marked present today: {self.attendance_manager.get_today_count()}")
        print(f" Press '{settings.REGISTER_KEY.upper()}' to register a new student.")
        print(f" Press '{settings.EXIT_KEY.upper()}' (or ESC) to exit safely.")
        print("=" * 60)

        try:
            while True:
                success, frame = self.capture.read()

                if not success or frame is None:
                    self.webcam_ok = False
                    print("[WARNING] Failed to read frame from webcam. Retrying...")
                    time.sleep(0.1)
                    continue

                self.webcam_ok = True

                if settings.FLIP_CAMERA:
                    frame = cv2.flip(frame, 1)

                frame = self._process_attendance_frame(frame)

                cv2.imshow(settings.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(settings.EXIT_KEY) or key == 27:  # 27 = ESC key
                    print("Exit key pressed. Shutting down safely.")
                    break
                if key == ord(settings.REGISTER_KEY):
                    self._register_new_student()

                if cv2.getWindowProperty(
                    settings.WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    print("Window closed. Shutting down safely.")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Shutting down safely.")
        except Exception as error:  # noqa: BLE001 - top-level safety net
            print(f"[ERROR] Unexpected failure: {error}")
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Attendance mode
    # ------------------------------------------------------------------
    def _process_attendance_frame(self, frame):
        """
        Runs detection + recognition (throttled to every N frames for
        performance), marks attendance for newly recognized students, and
        renders the full HUD overlay.
        """
        self._frame_counter += 1
        run_detection = (
            self._frame_counter % settings.PROCESS_EVERY_N_FRAMES == 0
            or not self._cached_results
        )

        if run_detection:
            rgb_small_frame, small_locations, original_locations = \
                self.face_detector.detect(frame)
            recognition_results = self.recognition_system.recognize(
                rgb_small_frame, small_locations
            )
            self._cached_results = list(zip(original_locations, recognition_results))

        best_confidence = 0.0

        for face_box, result in self._cached_results:
            best_confidence = max(best_confidence, result.confidence)

            if result.is_known:
                newly_marked = self.attendance_manager.mark_attendance(result.name)
                if newly_marked:
                    status_label, box_color = "MARKED PRESENT", settings.COLOR_SUCCESS
                else:
                    status_label, box_color = "ALREADY MARKED", settings.COLOR_WARNING
            else:
                status_label, box_color = "UNKNOWN", settings.COLOR_ERROR

            draw_face_box(frame, face_box, result.name, result.confidence,
                          status_label, box_color)

        if settings.SHOW_STATUS_PANEL:
            fps = self.fps_counter.update()
            frame = draw_status_panel(
                frame,
                fps=fps,
                webcam_ok=self.webcam_ok,
                faces_detected=len(self._cached_results),
                best_confidence=best_confidence,
                students_registered=len(self.recognition_system.known_encodings),
                attendance_today=self.attendance_manager.get_today_count(),
            )

        frame = draw_control_hints(frame)
        return frame

    # ------------------------------------------------------------------
    # Registration mode
    # ------------------------------------------------------------------
    def _register_new_student(self) -> None:
        """
        Interactive registration flow: prompts for a student name in the
        console, then captures a series of face images from the live
        webcam feed, saving each one and rebuilding the recognition
        system's encoding database once capture is complete.
        """
        print("\n=== New Student Registration ===")
        raw_name = input("Enter the student's full name: ").strip()

        if not raw_name:
            print("[WARNING] Registration cancelled: name cannot be empty.\n")
            return

        try:
            safe_name = self.dataset_manager.sanitize_name(raw_name)
        except ValueError as error:
            print(f"[WARNING] Registration cancelled: {error}\n")
            return

        if self.dataset_manager.student_exists(safe_name):
            print(f"[INFO] '{raw_name}' is already registered. Additional "
                  "samples will be added to improve recognition accuracy.")

        print(f"Capturing {settings.IMAGES_PER_REGISTRATION} face samples for "
              f"'{raw_name}'. Look directly at the camera.")
        print("Press ESC at any time to cancel registration.\n")

        captured = 0
        last_capture_time = 0.0

        while captured < settings.IMAGES_PER_REGISTRATION:
            success, frame = self.capture.read()
            if not success or frame is None:
                continue

            if settings.FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            _, _, face_boxes = self.face_detector.detect(frame)

            instruction = f"Capturing {captured}/{settings.IMAGES_PER_REGISTRATION} - Face the camera"
            box_color = settings.COLOR_SUCCESS

            if len(face_boxes) == 0:
                box_color = settings.COLOR_WARNING
                instruction = "No face detected - please face the camera"
            elif len(face_boxes) > 1:
                box_color = settings.COLOR_ERROR
                instruction = "Multiple faces detected - only one person should be in frame"
            else:
                now = time.time()
                if now - last_capture_time >= settings.REGISTRATION_CAPTURE_DELAY:
                    self.dataset_manager.save_face_image(safe_name, frame, captured)
                    captured += 1
                    last_capture_time = now

            for (top, right, bottom, left) in face_boxes:
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)

            cv2.putText(display_frame, instruction, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.COLOR_TEXT, 2)
            cv2.putText(display_frame, "Press ESC to cancel registration",
                        (12, display_frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings.COLOR_TEXT, 1)

            cv2.imshow(settings.WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[INFO] Registration cancelled by user.\n")
                return

        print(f"[SUCCESS] Captured {captured} image(s) for '{raw_name}'. "
              "Rebuilding face encoding database...")
        self.recognition_system.refresh()
        print(f"[SUCCESS] Registration complete. "
              f"'{raw_name}' can now be recognized for attendance.\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """Releases all camera and window resources."""
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        print("Resources released. Goodbye!")


if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.run()
