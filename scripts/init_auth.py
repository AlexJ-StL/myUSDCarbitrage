#!/usr/bin/env python3
"""CLI script to initialize the authentication system."""

import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.init_auth import initialize_auth_system

if __name__ == "__main__":
    initialize_auth_system()
