"""
YouTube data collector for compliance analysis
"""
import re
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import logging
from typing import Dict, Any, Optional, Tuple
import os
import time
from src.rate_limiter import youtube_rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeAnalyzer:
    def __init__(self, youtube_api_key=None):
        self.youtube_api_key = youtube_api_key
        
        # More aggressive rate limiting to prevent IP blocking
        self.max_concurrent_requests = int(os.getenv("YOUTUBE_MAX_CONCURRENT", "2"))  # Reduced from 5 to 2
        self.request_delay_seconds = float(os.getenv("YOUTUBE_REQUEST_DELAY", "3.0"))  # Increased from 2.0 to 3.0 seconds
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.last_request_time = 0
        self.rate_limit_lock = asyncio.Lock()
        
        logger.info(f"🚦 YouTube Rate Limiting Config:")
        logger.info(f"   └─ Max concurrent requests: {self.max_concurrent_requests}")
        logger.info(f"   └─ Delay between transcript requests: {self.request_delay_seconds}s")
        
    def extract_channel_info_from_url(self, url):
        """Extract channel ID, name, and handle from a YouTube URL"""
        try:
            # Parse URL
            parsed_url = urlparse(url)
            channel_id = None
            channel_name = None
            channel_handle = None
            
            # Channel URL pattern: youtube.com/channel/UC...
            if 'youtube.com/channel/' in url:
                channel_id = parsed_url.path.split('/channel/')[1].split('/')[0]
                # Always try to get channel info from the page
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get channel name from page title
                title_tag = soup.find('title')
                if title_tag:
                    channel_name = title_tag.text.replace(' - YouTube', '').strip()
                
                # Try to get channel handle from the page
                for link in soup.find_all('link'):
                    if link.get('rel') == ['canonical']:
                        canonical_url = link.get('href', '')
                        if '/@' in canonical_url:
                            channel_handle = '@' + canonical_url.split('/@')[1].split('/')[0]
                            break
                
                # If we have API key, also try to get info from API
                if self.youtube_api_key and not channel_name:
                    channel_info = self._get_channel_info_from_api(channel_id)
                    if channel_info:
                        channel_name = channel_info.get('title')
                        if not channel_handle:
                            channel_handle = channel_info.get('customUrl')
            
            # User URL pattern: youtube.com/user/username or youtube.com/@handle
            elif 'youtube.com/user/' in url or 'youtube.com/c/' in url or 'youtube.com/@' in url:
                # Extract handle from URL
                if 'youtube.com/@' in url:
                    channel_handle = '@' + parsed_url.path.split('/@')[1].split('/')[0]
                elif 'youtube.com/c/' in url:
                    channel_handle = '@' + parsed_url.path.split('/c/')[1].split('/')[0]
                elif 'youtube.com/user/' in url:
                    channel_handle = '@' + parsed_url.path.split('/user/')[1].split('/')[0]
                
                # Need to fetch the page and extract channel ID
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for channel ID in meta tags or other elements
                for link in soup.find_all('link'):
                    if 'channel_id' in link.get('href', ''):
                        channel_id = link.get('href').split('channel_id=')[1]
                        break
                
                # Get channel name from page title
                title_tag = soup.find('title')
                if title_tag:
                    channel_name = title_tag.text.replace(' - YouTube', '')
            
            # For video URLs, get channel from the video page
            elif 'youtube.com/watch' in url:
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
                if video_id:
                    channel_info = self.get_channel_info_from_video(video_id)
                    if channel_info:
                        channel_id = channel_info.get('channel_id')
                        channel_name = channel_info.get('channel_name')
                        channel_handle = channel_info.get('channel_handle')
                    
            if not channel_id:
                logger.warning(f"Couldn't extract channel ID from URL: {url}")
                return None, None, None
                
            return channel_id, channel_name, channel_handle
            
        except Exception as e:
            logger.error(f"Error extracting channel info: {str(e)}")
            return None, None, None
    
    def _get_channel_info_from_api(self, channel_id):
        """Get channel info using YouTube API"""
        if not self.youtube_api_key:
            return None
            
        try:
            # Track API call
            youtube_rate_limiter['total_api_calls'] += 1
            
            url = "https://www.googleapis.com/youtube/v3/channels"
            params = {
                "key": self.youtube_api_key,
                "id": channel_id,
                "part": "snippet"
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'items' in data and data['items']:
                snippet = data['items'][0]['snippet']
                return {
                    'title': snippet.get('title'),
                    'customUrl': snippet.get('customUrl')
                }
            return None
        except Exception as e:
            logger.error(f"Error getting channel info from API: {str(e)}")
            return None
    
    def get_channel_info_from_video(self, video_id):
        """Get channel info from a video ID by scraping the video page"""
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.debug(f"Fetching video page: {url}")
            
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for bad status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            channel_info = {}
            
            # Method 1: Look for channel ID in meta tags
            for meta in soup.find_all('meta'):
                if meta.get('itemprop') == 'channelId':
                    channel_info['channel_id'] = meta.get('content')
                    break
            
            # Method 2: Look for channel link
            if not channel_info.get('channel_id'):
                channel_link = soup.find('a', {'class': 'yt-simple-endpoint style-scope yt-formatted-string'})
                if channel_link:
                    href = channel_link.get('href', '')
                    if '/channel/' in href:
                        channel_info['channel_id'] = href.split('/channel/')[1].split('/')[0]
                    elif '/@' in href:
                        channel_info['channel_handle'] = href.split('/@')[1].split('/')[0]
                    elif '/c/' in href:
                        channel_info['channel_handle'] = href.split('/c/')[1].split('/')[0]
                    channel_info['channel_name'] = channel_link.text.strip()
            
            # Method 3: Look for channel ID in script tags
            if not channel_info.get('channel_id'):
                for script in soup.find_all('script'):
                    script_text = str(script)
                    if 'channelId' in script_text:
                        # Try to find channelId in the script
                        match = re.search(r'"channelId":"([^"]+)"', script_text)
                        if match:
                            channel_info['channel_id'] = match.group(1)
                            break
            
            # Method 4: Look for channel name in page title
            if not channel_info.get('channel_name'):
                title_tag = soup.find('title')
                if title_tag:
                    # YouTube titles are usually "Video Title - Channel Name - YouTube"
                    title_parts = title_tag.text.split(' - ')
                    if len(title_parts) >= 2:
                        channel_info['channel_name'] = title_parts[-2]
            
            if not channel_info.get('channel_id'):
                logger.error(f"Could not find channel ID in video page: {url}")
                return None
                
            logger.info(f"Successfully extracted channel info: {channel_info}")
            return channel_info
            
        except requests.RequestException as e:
            logger.error(f"Error fetching video page: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting channel info: {str(e)}")
            return None
    
    def _get_videos_by_scraping(self, channel_id, limit):
        """Get videos by scraping channel page"""
        try:
            url = f"https://www.youtube.com/channel/{channel_id}/videos"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            videos = []
            script_tags = soup.find_all('script')
            
            for script in script_tags:
                if 'var ytInitialData' in str(script):
                    json_text = str(script).split('var ytInitialData = ')[1].split(';</script>')[0]
                    
                    # Updated regex patterns to match YouTube's current structure
                    video_matches = re.findall(
                        r'"videoId":"([^"]+)".*?"title":{"runs":\[{"text":"([^"]+)"',
                        json_text
                    )
                    
                    # Alternative pattern if the first one doesn't match
                    if not video_matches:
                        video_matches = re.findall(
                            r'"videoId":"([^"]+)".*?"text":"([^"]+)"',
                            json_text
                        )
                    
                    # Another alternative pattern
                    if not video_matches:
                        video_matches = re.findall(
                            r'"videoId":"([^"]+)".*?"title":{"simpleText":"([^"]+)"',
                            json_text
                        )
                    
                    for i, (video_id, title) in enumerate(video_matches):
                        if i >= limit:
                            break
                        videos.append({
                            "id": video_id,
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}"
                        })
                    break
            
            # If no videos found with the above patterns, try a different approach
            if not videos:
                # Look for video elements directly
                video_elements = soup.find_all('a', {'id': 'video-title'})
                for i, element in enumerate(video_elements):
                    if i >= limit:
                        break
                    video_id = element.get('href', '').split('v=')[1].split('&')[0]
                    title = element.get('title', '')
                    if video_id and title:
                        videos.append({
                            "id": video_id,
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}"
                        })
            
            logger.info(f"Found {len(videos)} videos through scraping")
            return videos
            
        except Exception as e:
            logger.error(f"Error scraping videos: {str(e)}")
            return []
            
    def get_videos_from_channel(self, channel_id, limit=10):
        """Get recent videos from a channel"""
        try:
            # If YouTube API key is available, use the API
            if self.youtube_api_key:
                videos = self._get_videos_from_api(channel_id, limit)
                if videos:
                    logger.info(f"Found {len(videos)} videos through API")
                    return videos
                logger.warning("API returned no videos, falling back to scraping")
            
            # Otherwise, or if API failed, scrape the channel page
            videos = self._get_videos_by_scraping(channel_id, limit)
            if not videos:
                logger.warning(f"No videos found for channel {channel_id} through any method")
            return videos
            
        except Exception as e:
            logger.error(f"Error getting videos from channel: {str(e)}")
            return []
            
    def _get_videos_from_api(self, channel_id, limit):
        """Get videos using YouTube API"""
        # Track API call
        youtube_rate_limiter['total_api_calls'] += 1
        
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "key": self.youtube_api_key,
            "channelId": channel_id,
            "part": "snippet",
            "order": "date",
            "maxResults": limit,
            "type": "video"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            videos.append({
                "id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })
            
        return videos
    
    async def get_transcript_async(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get transcript for a YouTube video asynchronously with aggressive rate limiting"""
        async with self.semaphore:  # Limit concurrent requests
            # Aggressive rate limiting to prevent IP blocking
            async with self.rate_limit_lock:
                current_time = time.time()
                
                # Ensure minimum delay between requests
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_delay_seconds:
                    sleep_time = self.request_delay_seconds - time_since_last
                    logger.debug(f"⏱️  Rate limiting: sleeping {sleep_time:.1f}s before transcript request for {video_id}")
                    await asyncio.sleep(sleep_time)
                
                self.last_request_time = time.time()
            
            try:
                logger.debug(f"🎬 Fetching transcript for video {video_id}")
                
                # Use aiohttp for async HTTP requests
                async with aiohttp.ClientSession() as session:
                    # Get transcript data - support multiple languages
                    transcript_data = await asyncio.to_thread(
                        YouTubeTranscriptApi.get_transcript,
                        video_id,
                        languages=[
                            'en', 'en-US', 'en-GB',  # English variants
                            'de', 'de-DE',            # German
                            'fr', 'fr-FR',            # French  
                            'it', 'it-IT',            # Italian
                            'sv', 'sv-SE',            # Swedish
                            'es', 'es-ES', 'es-MX',   # Spanish (Spain and Mexico)
                            'nl', 'nl-NL'             # Dutch
                        ]
                    )
                    
                    if not transcript_data:
                        return None
                        
                    # Create full text from transcript segments
                    full_text = ' '.join([entry.get('text', '') for entry in transcript_data])
                    
                    logger.debug(f"✅ Successfully fetched transcript for {video_id} ({len(full_text)} chars)")
                    
                    # Add small additional delay after successful request to be extra safe
                    await asyncio.sleep(0.5)
                    
                    return {
                        'full_text': full_text,
                        'segments': transcript_data
                    }
            except (TranscriptsDisabled, NoTranscriptFound):
                # Hide verbose logging for common transcript failures
                logger.debug(f"No transcript available for video {video_id}")
                return None
            except Exception as e:
                # Only log errors for unexpected failures, not common transcript issues
                if "blocking requests" in str(e).lower() or "rate limit" in str(e).lower():
                    logger.debug(f"YouTube rate limiting detected for video {video_id}")
                    return {'error': str(e)}
                else:
                    # Hide most transcript errors as they're very common
                    logger.debug(f"Transcript fetch failed for video {video_id}: {str(e)}")
                return None

    def normalize_url(self, url: str) -> Tuple[str, str]:
        """
        Normalize a YouTube URL to a channel URL.
        If input is a video URL, extracts the channel URL.
        If input is already a channel URL, returns it unchanged.
        
        Returns:
            Tuple[str, str]: (normalized_url, url_type) where url_type is either 'channel' or 'video'
        """
        try:
            # Check if it's a video URL
            if 'youtube.com/watch' in url or 'youtu.be/' in url:
                logger.info(f"Input URL is a video URL: {url}")
                # Extract video ID
                if 'youtube.com/watch' in url:
                    video_id = parse_qs(urlparse(url).query).get('v', [None])[0]
                else:  # youtu.be/
                    video_id = urlparse(url).path.lstrip('/')
                
                if not video_id:
                    logger.error(f"Could not extract video ID from URL: {url}")
                    return url, 'unknown'
                
                # Get channel info using our existing method
                channel_info = self.get_channel_info_from_video(video_id)
                if not channel_info or not channel_info.get('channel_id'):
                    logger.error(f"Could not get channel info for video: {video_id}")
                    return url, 'unknown'
                
                # Construct channel URL
                channel_url = f"https://www.youtube.com/channel/{channel_info['channel_id']}"
                logger.info(f"Extracted channel URL: {channel_url}")
                return channel_url, 'video'
            
            # Check if it's a channel URL
            if any(pattern in url for pattern in ['youtube.com/channel/', 'youtube.com/c/', 'youtube.com/@', 'youtube.com/user/']):
                logger.info(f"Input URL is already a channel URL: {url}")
                return url, 'channel'
            
            logger.warning(f"Unrecognized URL format: {url}")
            return url, 'unknown'
            
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {str(e)}")
            return url, 'unknown'

    async def analyze_channel_async(self, url: str, video_limit: int = 10, use_concurrent: bool = False) -> Optional[Dict[str, Any]]:
        """
        Analyze a channel by collecting transcripts from its videos asynchronously.
        Can handle both channel URLs and video URLs.
        
        Args:
            url: URL of the YouTube channel or video
            video_limit: Maximum number of videos to analyze
            use_concurrent: Whether to use concurrent processing (for OpenAI)
        """
        # Normalize URL if it's a video URL
        normalized_url, url_type = self.normalize_url(url)
        if url_type == 'video':
            logger.info(f"Converting video URL to channel URL for analysis")
        
        channel_id, channel_name, channel_handle = self.extract_channel_info_from_url(normalized_url)
        if not channel_id:
            logger.error(f"Could not extract channel ID from {normalized_url}")
            return None
            
        videos = self.get_videos_from_channel(channel_id, limit=video_limit)
        logger.info(f"Found {len(videos)} videos for analysis")
        
        channel_data = {
            'channel_id': channel_id,
            'channel_name': channel_name,
            'channel_handle': channel_handle,
            'videos': []
        }
        
        if use_concurrent:
            logger.info("Using CONCURRENT processing for transcript fetching")
            # Create tasks for concurrent transcript fetching
            tasks = []
            for video in videos:
                logger.debug(f"Creating task for video: {video['id']} - {video['title']}")
                task = self.get_transcript_async(video['id'])
                tasks.append((video, task))
            
            # Wait for all transcript fetches to complete
            for video, task in tasks:
                logger.debug(f"Waiting for transcript: {video['id']} - {video['title']}")
                transcript = await task
                if transcript:
                    logger.info(f"✓ Successfully fetched transcript for video: {video['id']} - {video['title']}")
                    video['transcript'] = transcript
                    channel_data['videos'].append(video)
                else:
                    logger.warning(f"✗ Failed to fetch transcript for video: {video['id']} - {video['title']}")
        else:
            logger.info("Using SEQUENTIAL processing for transcript fetching")
            # Sequential processing for local LLM
            for video in videos:
                logger.debug(f"Fetching transcript for video: {video['id']} - {video['title']}")
                transcript = await self.get_transcript_async(video['id'])
                if transcript:
                    logger.info(f"✓ Successfully fetched transcript for video: {video['id']} - {video['title']}")
                    video['transcript'] = transcript
                    channel_data['videos'].append(video)
                else:
                    logger.warning(f"✗ Failed to fetch transcript for video: {video['id']} - {video['title']}")
                
        logger.info(f"Successfully retrieved transcripts for {len(channel_data['videos'])} videos")
        return channel_data

    # Keep the synchronous version for backward compatibility
    def analyze_channel(self, channel_url: str, video_limit: int = 10, use_concurrent: bool = False) -> Optional[Dict[str, Any]]:
        """Synchronous version of analyze_channel for backward compatibility"""
        return asyncio.run(self.analyze_channel_async(channel_url, video_limit, use_concurrent))

    # Keep the synchronous version for backward compatibility
    def get_transcript(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_transcript for backward compatibility"""
        return asyncio.run(self.get_transcript_async(video_id)) 