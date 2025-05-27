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
    'backoff_seconds': 60,  # Initial backoff period
    'max_backoff_seconds': 300,  # Max 5 minutes
    'consecutive_blocks': 0,
    'last_block_time': None,
    'lock': asyncio.Lock()
} 