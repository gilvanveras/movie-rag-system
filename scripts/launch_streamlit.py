#!/usr/bin/env python3
"""Setup and launch Streamlit with sample data."""

import os
import subprocess
import sys


def setup_and_launch():
    """Setup sample data and launch Streamlit."""
    print("🚀 Movie RAG System - Quick Setup & Launch")
    print("=" * 50)

    # Step 1: Add sample data
    print("1. Adding sample movie data...")
    try:
        result = subprocess.run(
            [sys.executable, "add_sample_data.py"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        if result.returncode == 0:
            print("✅ Sample data added successfully!")
        else:
            print(f"⚠️  Sample data script output: {result.stdout}")
            print(f"⚠️  Errors: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not add sample data: {e}")

    # Step 2: Launch Streamlit
    print("\n2. Launching Streamlit...")
    print("🌐 Opening browser to http://localhost:8501")
    print("📝 Try searching for: 'The Matrix' or 'Inception'")
    print("❓ Or click on movies in the 'Movies in Database' section")
    print("\n" + "=" * 50)

    try:
        # Launch Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/app.py"], cwd="."
        )
    except KeyboardInterrupt:
        print("\n👋 Streamlit stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n💡 Try manually running: streamlit run src/app.py")


if __name__ == "__main__":
    setup_and_launch()
