"""
Version information for YouTube Content Compliance Analyzer
"""

__version__ = "v1.0.1"
__version_info__ = (1, 0, 1)
__build_date__ = "2024-01-15"

# Version history
VERSION_HISTORY = {
    "v1.0.1": {
        "date": "2024-01-15",
        "changes": [
            "Fixed critical bug where URLs were lost in pipeline due to duplicate task_done() calls",
            "Added YouTube rate limiting with exponential backoff",
            "Added controversy pre-screening for creators",
            "Added PASS/FAIL/ERROR status column to CSV exports",
            "Improved error handling and logging",
            "Added retry limits for transcript fetching",
            "Added missing URL detection and recovery"
        ]
    },
    "v1.0.0": {
        "date": "2024-01-14",
        "changes": [
            "Initial release",
            "Video-based pipeline architecture",
            "Bulk analysis support",
            "CSV export functionality"
        ]
    }
} 