"""
YouTube data collector for compliance analysis
"""
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeAnalyzer:
    def __init__(self, youtube_api_key=None):
        self.youtube_api_key = youtube_api_key
        
    def extract_channel_id_from_url(self, url):
        """Extract channel ID from a YouTube URL"""
        try:
            # Parse URL
            parsed_url = urlparse(url)
            
            # Channel URL pattern: youtube.com/channel/UC...
            if 'youtube.com/channel/' in url:
                return parsed_url.path.split('/channel/')[1].split('/')[0]
            
            # User URL pattern: youtube.com/user/username
            elif 'youtube.com/user/' in url or 'youtube.com/c/' in url or 'youtube.com/@' in url:
                # Need to fetch the page and extract channel ID
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for channel ID in meta tags or other elements
                for link in soup.find_all('link'):
                    if 'channel_id' in link.get('href', ''):
                        return link.get('href').split('channel_id=')[1]
            
            # For video URLs, get channel from the video page
            elif 'youtube.com/watch' in url:
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
                if video_id:
                    return self.get_channel_id_from_video(video_id)
                    
            logger.warning(f"Couldn't extract channel ID from URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting channel ID: {str(e)}")
            return None
    
    def get_channel_id_from_video(self, video_id):
        """Get channel ID from a video ID by scraping the video page"""
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for channel link in the page
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if '/channel/' in href:
                    return href.split('/channel/')[1].split('/')[0]
                    
            return None
        except Exception as e:
            logger.error(f"Error getting channel from video: {str(e)}")
            return None
            
    def get_videos_from_channel(self, channel_id, limit=10):
        """Get recent videos from a channel"""
        try:
            # If YouTube API key is available, use the API
            if self.youtube_api_key:
                return self._get_videos_from_api(channel_id, limit)
            
            # Otherwise, scrape the channel page
            return self._get_videos_by_scraping(channel_id, limit)
        except Exception as e:
            logger.error(f"Error getting videos from channel: {str(e)}")
            return []
            
    def _get_videos_from_api(self, channel_id, limit):
        """Get videos using YouTube API"""
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
        
    def _get_videos_by_scraping(self, channel_id, limit):
        """Get videos by scraping channel page"""
        url = f"https://www.youtube.com/channel/{channel_id}/videos"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        videos = []
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if 'var ytInitialData' in str(script):
                json_text = str(script).split('var ytInitialData = ')[1].split(';</script>')[0]
                
                # Extract video IDs using regex
                video_matches = re.findall(r'"videoRenderer":{"videoId":"([^"]+)","thumbnail".+?"text":"([^"]+)"', json_text)
                
                for i, (video_id, title) in enumerate(video_matches):
                    if i >= limit:
                        break
                    videos.append({
                        "id": video_id,
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    })
                break
                
        return videos
    
    def get_transcript(self, video_id):
        """Get transcript for a YouTube video"""
        try:
            # Directly get transcript data - simpler approach to avoid the error
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            
            if not transcript_data:
                return None
                
            # Create full text from transcript segments
            full_text = ' '.join([entry.get('text', '') for entry in transcript_data])
            
            return {
                'full_text': full_text,
                'segments': transcript_data
            }
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"No transcript available for video {video_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {str(e)}")
            return None
            
    def analyze_channel(self, channel_url, video_limit=10):
        """Analyze a channel by collecting transcripts from its videos"""
        channel_id = self.extract_channel_id_from_url(channel_url)
        if not channel_id:
            logger.error(f"Could not extract channel ID from {channel_url}")
            return None
            
        videos = self.get_videos_from_channel(channel_id, limit=video_limit)
        logger.info(f"Found {len(videos)} videos for analysis")
        
        channel_data = {
            'channel_id': channel_id,
            'videos': []
        }
        
        for video in videos:
            transcript = self.get_transcript(video['id'])
            if transcript:
                video['transcript'] = transcript
                channel_data['videos'].append(video)
                
        logger.info(f"Successfully retrieved transcripts for {len(channel_data['videos'])} videos")
        return channel_data 