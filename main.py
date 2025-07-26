#!/usr/bin/env python3
"""Main entry point for the Movie RAG System Streamlit application."""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"])
