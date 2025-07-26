#!/usr/bin/env python3
"""Main entry point for the Movie RAG System Streamlit application."""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and run the main function from app.py
try:
    from app import main
    main()
except ImportError as e:
    print(f"Error importing app: {e}")
    print("Make sure you're running this with: streamlit run main.py")
