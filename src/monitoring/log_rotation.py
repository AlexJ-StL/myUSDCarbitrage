"""Log rotation and archiving utilities."""

import asyncio
import gzip
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# Configure module logger
logger = logging.getLogger(__name__)


class LogRotator:
    """Handles log rotation and archiving."""

    def __init__(
        self,
        log_dir: str = "logs",
        max_size_mb: int = 100,
        backup_count: int = 5,
        compress: bool = True,
        archive_dir: Optional[str] = None,
    ):
        """Initialize log rotator."""
        self.log_dir = log_dir
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
        self.compress = compress
        self.archive_dir = archive_dir or os.path.join(log_dir, "archive")

        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        if self.archive_dir:
            os.makedirs(self.archive_dir, exist_ok=True)

    def check_and_rotate_logs(self) -> List[str]:
        """Check log files and rotate if needed."""
        rotated_files = []

        try:
            # Get all log files in the directory
            log_files = [
                f
                for f in os.listdir(self.log_dir)
                if f.endswith(".log") and os.path.isfile(os.path.join(self.log_dir, f))
            ]

            for log_file in log_files:
                file_path = os.path.join(self.log_dir, log_file)

                # Check file size
                size_mb = os.path.getsize(file_path) / (1024 * 1024)

                if size_mb >= self.max_size_mb:
                    rotated_path = self._rotate_log(file_path)
                    if rotated_path:
                        rotated_files.append(rotated_path)
                        logger.info(
                            f"Rotated log file {log_file} ({size_mb:.2f}MB)",
                            extra={
                                "component": "log_rotation",
                                "file": log_file,
                                "size_mb": size_mb,
                                "rotated_path": rotated_path,
                            },
                        )

            return rotated_files

        except Exception as e:
            logger.error(
                f"Error checking and rotating logs: {e}",
                exc_info=True,
                extra={"component": "log_rotation"},
            )
            return []

    def _rotate_log(self, file_path: str) -> Optional[str]:
        """Rotate a single log file."""
        try:
            base_name = os.path.basename(file_path)
            name_parts = os.path.splitext(base_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create rotated file name
            rotated_name = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
            rotated_path = os.path.join(self.log_dir, rotated_name)

            # Rename current log file
            shutil.copy2(file_path, rotated_path)

            # Truncate original file
            with open(file_path, "w") as f:
                f.write(f"Log rotated at {datetime.now().isoformat()}\n")

            # Compress if needed
            if self.compress:
                compressed_path = f"{rotated_path}.gz"
                with open(rotated_path, "rb") as f_in:
                    with gzip.open(compressed_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove uncompressed rotated file
                os.remove(rotated_path)
                rotated_path = compressed_path

            # Archive if needed
            if self.archive_dir:
                archive_path = os.path.join(
                    self.archive_dir, os.path.basename(rotated_path)
                )
                shutil.move(rotated_path, archive_path)
                rotated_path = archive_path

            # Remove old backups if needed
            self._remove_old_backups(name_parts[0])

            return rotated_path

        except Exception as e:
            logger.error(
                f"Error rotating log file {file_path}: {e}",
                exc_info=True,
                extra={"component": "log_rotation", "file": file_path},
            )
            return None

    def _remove_old_backups(self, base_name: str):
        """Remove old backup files if there are more than backup_count."""
        try:
            # Get all backup files for this base name
            backup_files = []

            # Check in log directory
            for f in os.listdir(self.log_dir):
                if f.startswith(base_name + "_") and (
                    f.endswith(".log") or f.endswith(".log.gz")
                ):
                    backup_files.append(os.path.join(self.log_dir, f))

            # Check in archive directory
            if self.archive_dir and os.path.exists(self.archive_dir):
                for f in os.listdir(self.archive_dir):
                    if f.startswith(base_name + "_") and (
                        f.endswith(".log") or f.endswith(".log.gz")
                    ):
                        backup_files.append(os.path.join(self.archive_dir, f))

            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x))

            # Remove oldest files if there are too many
            while len(backup_files) > self.backup_count:
                oldest = backup_files.pop(0)
                os.remove(oldest)
                logger.info(
                    f"Removed old log backup: {oldest}",
                    extra={"component": "log_rotation", "file": oldest},
                )

        except Exception as e:
            logger.error(
                f"Error removing old backups for {base_name}: {e}",
                exc_info=True,
                extra={"component": "log_rotation", "base_name": base_name},
            )

    async def start_rotation_scheduler(self, check_interval_minutes: int = 60):
        """Start a background task to periodically check and rotate logs."""
        logger.info(
            f"Starting log rotation scheduler (interval: {check_interval_minutes} minutes)",
            extra={
                "component": "log_rotation",
                "interval_minutes": check_interval_minutes,
            },
        )

        while True:
            try:
                self.check_and_rotate_logs()
                await asyncio.sleep(check_interval_minutes * 60)
            except asyncio.CancelledError:
                logger.info(
                    "Log rotation scheduler cancelled",
                    extra={"component": "log_rotation"},
                )
                break
            except Exception as e:
                logger.error(
                    f"Error in log rotation scheduler: {e}",
                    exc_info=True,
                    extra={"component": "log_rotation"},
                )
                await asyncio.sleep(60)  # Wait a minute before retrying

    def archive_old_logs(self, days: int = 30) -> int:
        """Archive logs older than specified days."""
        try:
            if not self.archive_dir:
                logger.warning(
                    "Archive directory not specified, cannot archive old logs",
                    extra={"component": "log_rotation"},
                )
                return 0

            # Ensure archive directory exists
            os.makedirs(self.archive_dir, exist_ok=True)

            # Calculate cutoff time
            cutoff_time = time.time() - (days * 24 * 3600)
            archived_count = 0

            # Get all log files in the directory
            log_files = [
                f
                for f in os.listdir(self.log_dir)
                if (f.endswith(".log") or f.endswith(".log.gz"))
                and os.path.isfile(os.path.join(self.log_dir, f))
            ]

            for log_file in log_files:
                file_path = os.path.join(self.log_dir, log_file)

                # Check file modification time
                if os.path.getmtime(file_path) < cutoff_time:
                    # Archive file
                    archive_path = os.path.join(self.archive_dir, log_file)

                    # Compress if not already compressed
                    if log_file.endswith(".log") and self.compress:
                        compressed_path = f"{file_path}.gz"
                        with open(file_path, "rb") as f_in:
                            with gzip.open(compressed_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        # Remove uncompressed file
                        os.remove(file_path)
                        file_path = compressed_path
                        archive_path = os.path.join(
                            self.archive_dir, os.path.basename(file_path)
                        )

                    # Move to archive
                    shutil.move(file_path, archive_path)
                    archived_count += 1

                    logger.info(
                        f"Archived old log file: {log_file}",
                        extra={
                            "component": "log_rotation",
                            "file": log_file,
                            "archive_path": archive_path,
                        },
                    )

            return archived_count

        except Exception as e:
            logger.error(
                f"Error archiving old logs: {e}",
                exc_info=True,
                extra={"component": "log_rotation", "days": days},
            )
            return 0


# Global log rotator instance
log_rotator = LogRotator()


def get_log_rotator() -> LogRotator:
    """Get log rotator instance."""
    return log_rotator


def configure_log_rotation(
    log_dir: str = "logs",
    max_size_mb: int = 100,
    backup_count: int = 5,
    compress: bool = True,
    archive_dir: Optional[str] = None,
) -> LogRotator:
    """Configure log rotation settings."""
    global log_rotator
    log_rotator = LogRotator(
        log_dir=log_dir,
        max_size_mb=max_size_mb,
        backup_count=backup_count,
        compress=compress,
        archive_dir=archive_dir,
    )
    return log_rotator
