"""
Video cache system for storing transcripts and LLM analysis results
"""
import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import hashlib

logger = logging.getLogger(__name__)

class VideoCache:
    def __init__(self, db_path: str = "data/video_cache.db"):
        self.db_path = db_path
        self.cache_expiry_days = 30
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Clean expired entries on startup
        self._cleanup_expired_entries()
        
        logger.info(f"📦 Video cache initialized at {db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_url TEXT UNIQUE NOT NULL,
                    video_id TEXT NOT NULL,
                    video_title TEXT,
                    channel_id TEXT,
                    channel_name TEXT,
                    transcript_json TEXT NOT NULL,
                    llm_result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    url_hash TEXT NOT NULL
                )
            ''')
            
            # Create index on video_url for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_video_url ON video_cache(video_url)
            ''')
            
            # Create index on url_hash for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_url_hash ON video_cache(url_hash)
            ''')
            
            # Create index on created_at for cleanup operations
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON video_cache(created_at)
            ''')
            
            conn.commit()
    
    def _get_url_hash(self, video_url: str) -> str:
        """Generate a hash for the video URL for faster lookups"""
        return hashlib.md5(video_url.encode()).hexdigest()
    
    def _cleanup_expired_entries(self):
        """Remove entries older than cache_expiry_days"""
        try:
            expiry_date = datetime.now() - timedelta(days=self.cache_expiry_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count expired entries before deletion
                cursor.execute('''
                    SELECT COUNT(*) FROM video_cache 
                    WHERE created_at < ?
                ''', (expiry_date.isoformat(),))
                
                expired_count = cursor.fetchone()[0]
                
                if expired_count > 0:
                    # Delete expired entries
                    cursor.execute('''
                        DELETE FROM video_cache 
                        WHERE created_at < ?
                    ''', (expiry_date.isoformat(),))
                    
                    conn.commit()
                    logger.info(f"🧹 Cleaned up {expired_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {str(e)}")
    
    def get_cached_result(self, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached transcript and LLM result for a video URL
        
        Returns:
            Dict with 'transcript' and 'llm_result' keys if found, None otherwise
        """
        try:
            url_hash = self._get_url_hash(video_url)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if entry exists and is not expired
                expiry_date = datetime.now() - timedelta(days=self.cache_expiry_days)
                
                cursor.execute('''
                    SELECT video_id, video_title, channel_id, channel_name, 
                           transcript_json, llm_result_json, created_at
                    FROM video_cache 
                    WHERE url_hash = ? AND created_at > ?
                ''', (url_hash, expiry_date.isoformat()))
                
                result = cursor.fetchone()
                
                if result:
                    # Update last_accessed timestamp
                    cursor.execute('''
                        UPDATE video_cache 
                        SET last_accessed = CURRENT_TIMESTAMP 
                        WHERE url_hash = ?
                    ''', (url_hash,))
                    conn.commit()
                    
                    video_id, video_title, channel_id, channel_name, transcript_json, llm_result_json, created_at = result
                    
                    # Parse JSON data
                    transcript = json.loads(transcript_json)
                    llm_result = json.loads(llm_result_json)
                    
                    logger.debug(f"✅ Cache hit for video {video_id}: {video_title}")
                    
                    return {
                        'video_id': video_id,
                        'video_title': video_title,
                        'video_url': video_url,
                        'channel_id': channel_id,
                        'channel_name': channel_name,
                        'transcript': transcript,
                        'llm_result': llm_result,
                        'cached_at': created_at,
                        'cache_hit': True
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached result for {video_url}: {str(e)}")
            return None
    
    def store_result(self, video_url: str, video_id: str, video_title: str, 
                    channel_id: str, channel_name: str, transcript: Dict[str, Any], 
                    llm_result: Dict[str, Any]) -> bool:
        """
        Store transcript and LLM result in cache
        
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            url_hash = self._get_url_hash(video_url)
            
            # Convert to JSON
            transcript_json = json.dumps(transcript)
            llm_result_json = json.dumps(llm_result)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace entry
                cursor.execute('''
                    INSERT OR REPLACE INTO video_cache 
                    (video_url, video_id, video_title, channel_id, channel_name, 
                     transcript_json, llm_result_json, url_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_url, video_id, video_title, channel_id, channel_name,
                      transcript_json, llm_result_json, url_hash))
                
                conn.commit()
                
                logger.debug(f"💾 Cached result for video {video_id}: {video_title}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing result in cache for {video_url}: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute('SELECT COUNT(*) FROM video_cache')
                total_entries = cursor.fetchone()[0]
                
                # Entries from last 24 hours
                yesterday = datetime.now() - timedelta(days=1)
                cursor.execute('''
                    SELECT COUNT(*) FROM video_cache 
                    WHERE created_at > ?
                ''', (yesterday.isoformat(),))
                recent_entries = cursor.fetchone()[0]
                
                # Oldest entry
                cursor.execute('''
                    SELECT MIN(created_at) FROM video_cache
                ''')
                oldest_entry = cursor.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'recent_entries_24h': recent_entries,
                    'oldest_entry': oldest_entry,
                    'cache_expiry_days': self.cache_expiry_days
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'total_entries': 0,
                'recent_entries_24h': 0,
                'oldest_entry': None,
                'cache_expiry_days': self.cache_expiry_days
            }
    
    def clear_cache(self) -> bool:
        """Clear all cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM video_cache')
                conn.commit()
                
                logger.info("🧹 Cache cleared successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

# Global cache instance
video_cache = VideoCache() 