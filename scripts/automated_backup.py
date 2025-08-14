#!/usr/bin/env python3
"""
Automated Database Backup Script

This script implements:
1. Scheduled database backups
2. Backup rotation (keeping only N most recent backups)
3. Optional cloud storage upload (AWS S3)
4. Backup verification
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta

import boto3
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("database_backup.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("automated_backup")

# Load environment variables
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "usdc_arbitrage"),
    "user": os.getenv("DB_USER", "arb_user"),
    "password": os.getenv("DB_PASSWORD", "strongpassword"),
    "host": os.getenv("DB_HOST", "localhost"),
}

# S3 configuration
S3_BUCKET = os.getenv("S3_BACKUP_BUCKET")
S3_PREFIX = os.getenv("S3_BACKUP_PREFIX", "database_backups/")


def create_backup(backup_dir="backups"):
    """Create a database backup."""
    logger.info("Creating database backup")

    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"usdc_arbitrage_{timestamp}.sql")

    # Create backup using pg_dump
    cmd = [
        "pg_dump",
        f"--dbname=postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}/{DB_PARAMS['dbname']}",
        "--format=custom",
        f"--file={backup_file}",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Backup created successfully: {backup_file}")
        return backup_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup failed: {e.stderr.decode()}")
        return None


def verify_backup(backup_file):
    """Verify backup integrity."""
    logger.info(f"Verifying backup integrity: {backup_file}")

    if not os.path.exists(backup_file):
        logger.error(f"Backup file not found: {backup_file}")
        return False

    # Verify backup using pg_restore --list
    cmd = ["pg_restore", "--list", backup_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        if result.returncode == 0 and result.stdout:
            logger.info(f"Backup verification successful: {backup_file}")
            return True
        else:
            logger.error(f"Backup verification failed: {backup_file}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Backup verification failed: {e.stderr.decode()}")
        return False


def upload_to_s3(backup_file):
    """Upload backup to S3."""
    if not S3_BUCKET:
        logger.info("S3 bucket not configured, skipping upload")
        return False

    logger.info(f"Uploading backup to S3: {backup_file}")

    try:
        s3_client = boto3.client("s3")
        filename = os.path.basename(backup_file)
        s3_key = f"{S3_PREFIX}{filename}"

        s3_client.upload_file(backup_file, S3_BUCKET, s3_key)
        logger.info(f"Backup uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload backup to S3: {e}")
        return False


def rotate_backups(backup_dir="backups", keep_count=7):
    """Delete old backups, keeping only the most recent ones."""
    logger.info(f"Rotating backups, keeping {keep_count} most recent")

    if not os.path.exists(backup_dir):
        logger.warning(f"Backup directory not found: {backup_dir}")
        return

    # List all backup files
    backup_files = []
    for file in os.listdir(backup_dir):
        if file.startswith("usdc_arbitrage_") and file.endswith(".sql"):
            file_path = os.path.join(backup_dir, file)
            backup_files.append((file_path, os.path.getmtime(file_path)))

    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x[1], reverse=True)

    # Delete old backups
    if len(backup_files) > keep_count:
        for file_path, _ in backup_files[keep_count:]:
            try:
                os.remove(file_path)
                logger.info(f"Deleted old backup: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete backup {file_path}: {e}")


def main():
    """Main function to run automated backup."""
    parser = argparse.ArgumentParser(description="Automated database backup")
    parser.add_argument(
        "--backup-dir", default="backups", help="Directory to store backups"
    )
    parser.add_argument(
        "--keep-count", type=int, default=7, help="Number of backups to keep"
    )
    parser.add_argument("--upload-s3", action="store_true", help="Upload backup to S3")
    parser.add_argument("--verify", action="store_true", help="Verify backup integrity")

    args = parser.parse_args()

    # Create backup
    backup_file = create_backup(args.backup_dir)
    if not backup_file:
        sys.exit(1)

    # Verify backup if requested
    if args.verify:
        if not verify_backup(backup_file):
            sys.exit(1)

    # Upload to S3 if requested
    if args.upload_s3:
        upload_to_s3(backup_file)

    # Rotate backups
    rotate_backups(args.backup_dir, args.keep_count)

    logger.info("Backup process completed successfully")


if __name__ == "__main__":
    main()
