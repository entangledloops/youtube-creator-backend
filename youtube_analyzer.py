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
                channel_info = self._get_channel_info_from_api(channel_id)
                if channel_info:
                    channel_name = channel_info.get('title')
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
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            channel_info = {}
            
            # Get channel ID
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if '/channel/' in href:
                    channel_info['channel_id'] = href.split('/channel/')[1].split('/')[0]
                    break
            
            # Get channel name and handle
            channel_link = soup.find('a', {'class': 'yt-simple-endpoint style-scope yt-formatted-string'})
            if channel_link:
                channel_info['channel_name'] = channel_link.text.strip()
                href = channel_link.get('href', '')
                if '/@' in href:
                    channel_info['channel_handle'] = href.split('/@')[1].split('/')[0]
                elif '/c/' in href:
                    channel_info['channel_handle'] = href.split('/c/')[1].split('/')[0]
            
            return channel_info if channel_info.get('channel_id') else None
            
        except Exception as e:
            logger.error(f"Error getting channel info from video: {str(e)}")
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
        channel_id, channel_name, channel_handle = self.extract_channel_info_from_url(channel_url)
        if not channel_id:
            logger.error(f"Could not extract channel ID from {channel_url}")
            return None
            
        videos = self.get_videos_from_channel(channel_id, limit=video_limit)
        logger.info(f"Found {len(videos)} videos for analysis")
        
        channel_data = {
            'channel_id': channel_id,
            'channel_name': channel_name,
            'channel_handle': channel_handle,
            'videos': []
        }
        
        for video in videos:
            transcript = self.get_transcript(video['id'])
            if transcript:
                video['transcript'] = transcript
                channel_data['videos'].append(video)
                
        logger.info(f"Successfully retrieved transcripts for {len(channel_data['videos'])} videos")
        return channel_data 