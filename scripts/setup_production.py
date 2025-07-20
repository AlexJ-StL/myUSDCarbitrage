#!/usr/bin/env python3
"""
Production setup script for USDC Arbitrage application.
This script prepares the environment for production deployment on a local machine.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_prerequisites():
    """Check if all required software is installed."""
    print("Checking