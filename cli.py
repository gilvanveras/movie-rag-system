#!/usr/bin/env python3
"""Command Line Interface for the Movie RAG System."""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    # Import and run the CLI
    from cli import main

    main()
