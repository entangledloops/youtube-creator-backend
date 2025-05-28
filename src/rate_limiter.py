"""
Shared rate limiting functionality for YouTube API and transcript requests
"""
import asyncio
import time

# Global YouTube rate limiting and tracking
youtube_rate_limiter = {
    'total_transcript_requests': 0,
    'total_api_calls': 0,
    'blocked_until': None,
    'backoff_seconds': 180,  # Increased from 120 to 180 seconds initial backoff
    'max_backoff_seconds': 900,  # Increased from 600 to 900 seconds (15 minutes max)
    'consecutive_blocks': 0,
    'last_block_time': None,
    'lock': asyncio.Lock()
} 