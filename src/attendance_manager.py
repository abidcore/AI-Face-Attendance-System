"""
src/attendance_manager.py
===========================
Owns all attendance record-keeping: initializing the attendance CSV file,
tracking which students have already been marked present today (to
prevent duplicate entries within the same session), and appending new
attendance rows.

Design notes
------------
- `pandas` is used for the relatively infrequent, bulk operations of
  creating the CSV with the correct headers and loading today's already-
  marked names at startup, since it comfortably handles CSV structure
  and date filtering.
- Python's built-in `csv` module is used for the actual per-event
  attendance write, since appending a single row on every recognition
  event is a hot path that does not need pandas' heavier DataFrame
  machinery - this keeps real-time recognition responsive.
"""

import csv
import os
from datetime import datetime
from typing import Set

import pandas as pd

from config import settings


class AttendanceManager:
    """Manages reading, writing, and duplicate-prevention for attendance records."""

    def __init__(self) -> None:
        self._ensure_csv_exists()
        self._marked_today: Set[str] = self._load_marked_today()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _ensure_csv_exists(self) -> None:
        """Creates the attendance CSV with proper headers if it doesn't exist."""
        os.makedirs(os.path.dirname(settings.ATTENDANCE_CSV_PATH), exist_ok=True)

        if not os.path.exists(settings.ATTENDANCE_CSV_PATH):
            empty_df = pd.DataFrame(columns=settings.ATTENDANCE_CSV_HEADERS)
            empty_df.to_csv(settings.ATTENDANCE_CSV_PATH, index=False)
            print(f"[INFO] Created new attendance file at "
                  f"'{settings.ATTENDANCE_CSV_PATH}'.")

    def _load_marked_today(self) -> Set[str]:
        """
        Loads the set of student names already marked present today,
        so a restarted session doesn't allow duplicate attendance for
        people who were already marked earlier the same day.
        """
        try:
            records = pd.read_csv(settings.ATTENDANCE_CSV_PATH)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return set()

        if records.empty or "Date" not in records.columns:
            return set()

        today = datetime.now().date().isoformat()
        todays_records = records[records["Date"] == today]
        return set(todays_records["Name"].tolist())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_already_marked(self, student_name: str) -> bool:
        """Returns True if the student has already been marked present today."""
        return student_name in self._marked_today

    def mark_attendance(self, student_name: str) -> bool:
        """
        Marks a student present for today, unless they have already been
        marked (duplicate-attendance prevention for the current session).

        Returns True if a new attendance record was written, False if the
        student was already marked today (no duplicate row written).
        """
        if self.is_already_marked(student_name):
            return False

        now = datetime.now()
        row = [student_name, now.date().isoformat(), now.strftime("%H:%M:%S"), "Present"]

        with open(settings.ATTENDANCE_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

        self._marked_today.add(student_name)
        return True

    def get_today_count(self) -> int:
        """Returns how many unique students have been marked present today."""
        return len(self._marked_today)

    def get_all_records(self) -> pd.DataFrame:
        """Returns the full attendance history as a pandas DataFrame."""
        try:
            return pd.read_csv(settings.ATTENDANCE_CSV_PATH)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame(columns=settings.ATTENDANCE_CSV_HEADERS)
