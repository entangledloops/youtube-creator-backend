#!/usr/bin/env python3
"""
Command-line tool for analyzing YouTube channels or videos
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv
import logging

from youtube_analyzer import YouTubeAnalyzer
from llm_analyzer import LLMAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def save_results(results, output_file=None):
    """Save analysis results to a file or print to console"""
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print(json.dumps(results, indent=2))

def analyze_channel(url, limit=5, provider="local", output=None):
    """Analyze a YouTube channel"""
    # Initialize analyzers
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    youtube_analyzer = YouTubeAnalyzer(youtube_api_key=youtube_api_key)
    llm_analyzer = LLMAnalyzer(provider=provider)
    
    # Get channel data
    logger.info(f"Analyzing channel: {url}")
    channel_data = youtube_analyzer.analyze_channel(url, video_limit=limit)
    
    if not channel_data:
        logger.error("Could not extract channel data")
        return
        
    if not channel_data.get('videos'):
        logger.warning("No videos with transcripts found for analysis")
        return
    
    # Analyze content
    logger.info("Analyzing content against compliance categories...")
    analysis_results = llm_analyzer.analyze_channel_content(channel_data)
    
    # Save or print results
    save_results(analysis_results, output)
    
    return analysis_results

def analyze_video(url, provider="local", output=None):
    """Analyze a single YouTube video"""
    # Initialize analyzers
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    youtube_analyzer = YouTubeAnalyzer(youtube_api_key=youtube_api_key)
    llm_analyzer = LLMAnalyzer(provider=provider)
    
    # Extract video ID
    video_id = None
    if "youtube.com/watch" in url and "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
        
    if not video_id:
        logger.error("Invalid YouTube video URL")
        return
    
    # Get video transcript
    logger.info(f"Getting transcript for video: {video_id}")
    transcript = youtube_analyzer.get_transcript(video_id)
    
    if not transcript:
        logger.error("No transcript available for this video")
        return
    
    # Get video title by scraping
    import requests
    from bs4 import BeautifulSoup
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').text.replace(' - YouTube', '')
    
    # Analyze transcript
    logger.info("Analyzing content against compliance categories...")
    analysis = llm_analyzer.analyze_transcript(
        transcript_text=transcript['full_text'],
        video_title=title,
        video_id=video_id
    )
    
    # Get channel ID
    channel_id = youtube_analyzer.get_channel_id_from_video(video_id)
    
    # Format results
    results = {
        "channel_id": channel_id,
        "video_analyses": [{
            "video_id": video_id,
            "video_title": title,
            "video_url": url,
            "analysis": analysis
        }]
    }
    
    # Save or print results
    save_results(results, output)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="YouTube Content Compliance Analyzer")
    
    # Create subparsers for channel and video commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Channel analysis command
    channel_parser = subparsers.add_parser("channel", help="Analyze a YouTube channel")
    channel_parser.add_argument("url", help="YouTube channel URL")
    channel_parser.add_argument("-l", "--limit", type=int, default=5, help="Number of videos to analyze (default: 5)")
    channel_parser.add_argument("-p", "--provider", choices=["local", "openai"], default="local", help="LLM provider (default: local)")
    channel_parser.add_argument("-o", "--output", help="Output file path (default: print to console)")
    
    # Video analysis command
    video_parser = subparsers.add_parser("video", help="Analyze a YouTube video")
    video_parser.add_argument("url", help="YouTube video URL")
    video_parser.add_argument("-p", "--provider", choices=["local", "openai"], default="local", help="LLM provider (default: local)")
    video_parser.add_argument("-o", "--output", help="Output file path (default: print to console)")
    
    args = parser.parse_args()
    
    if args.command == "channel":
        analyze_channel(args.url, args.limit, args.provider, args.output)
    elif args.command == "video":
        analyze_video(args.url, args.provider, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 