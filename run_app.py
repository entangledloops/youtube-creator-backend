#!/usr/bin/env python3
"""
Entry point for the YouTube Content Compliance Analyzer
"""
import sys
import os
import uvicorn

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Run the FastAPI app using import string for proper reload support
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 