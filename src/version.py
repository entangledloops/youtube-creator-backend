"""
Version information for YouTube Content Compliance Analyzer
"""

__version__ = "1.0.3"
__version_info__ = (1, 0, 3)
__build_date__ = "2025-05-27"

# Version history
VERSION_HISTORY = {
    "1.0.3": {
        "date": "2025-05-27",
        "changes": [
            "Fixed YouTube API call tracking - now properly counts all API calls made",
            "Fixed channel name extraction for channel ID URLs to ensure controversy screening works",
            "Added detailed debug logging for controversy screening to track LLM responses",
            "Improved channel info extraction to always get channel name even from direct channel ID URLs",
            "Fixed import issue for API call tracking in youtube_analyzer.py"
        ]
    },
    "1.0.2": {
        "date": "2025-05-27",
        "changes": [
            "Added separate controversy screening queue in the pipeline for better tracking and error handling",
            "Fixed controversy screening to properly distinguish between flagged creators (fail) vs screening errors (continue)",
            "Added controversy_check_failures tracking to identify channels processed despite screening errors",
            "Improved pipeline stage tracking with new stages: queued_for_controversy, screening_controversy, controversy_check_failed",
            "Enhanced single creator analysis to include controversy screening before processing videos",
            "Fixed worker cleanup to properly await cancelled tasks and prevent 'Task was destroyed' warnings",
            "Fixed variable name collision between result_worker function and result_worker_task variable",
            "Made controversy screening more specific to flag only current, ongoing controversies",
            "Added debug logging throughout the pipeline for better troubleshooting"
        ]
    },
    "1.0.1": {
        "date": "2025-05-25",
        "changes": [
            "Fixed duplicate task_done() calls in channel discovery worker",
            "Added proper error handling for invalid channel URLs",
            "Improved queue management to prevent synchronization issues",
            "Added debug logging for queue operations",
            "Fixed 'task_done() called too many times' error"
        ]
    },
    "1.0.0": {
        "date": "2025-05-23",
        "initial_release": True,
        "features": [
            "YouTube channel and video analysis",
            "Content compliance checking across multiple categories",
            "Bulk analysis with CSV upload",
            "Real-time progress tracking",
            "Export results as CSV or JSON evidence",
            "Support for OpenAI and local LLM providers",
            "Queue-based processing pipeline",
            "Controversy pre-screening for creators"
        ]
    }
} 