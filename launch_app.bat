@echo off
echo 🚀 Launching Movie RAG System Streamlit App
echo ============================================
echo.
echo The app will open in your default browser at http://localhost:8501
echo.
echo Available features:
echo • Search and analyze movies (The Matrix, Inception already loaded)
echo • Ask questions about movie reviews and themes
echo • View sentiment analysis and ratings
echo.
echo Press Ctrl+C to stop the server
echo.
streamlit run src/app.py
