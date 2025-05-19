"""
FastAPI application for YouTube content compliance analysis
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import logging
import uvicorn
import asyncio

from youtube_analyzer import YouTubeAnalyzer
from llm_analyzer import LLMAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Content Compliance Analyzer",
    description="API for analyzing YouTube content against compliance categories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation models
class ChannelAnalysisRequest(BaseModel):
    channel_url: HttpUrl
    video_limit: Optional[int] = 5
    llm_provider: Optional[str] = "local"  # "local" or "openai"

class VideoAnalysisRequest(BaseModel):
    video_url: HttpUrl
    llm_provider: Optional[str] = "local"  # "local" or "openai"

class MultipleURLsRequest(BaseModel):
    urls: List[str]
    llm_provider: Optional[str] = "local"  # "local" or "openai"

# Response models
class ErrorResponse(BaseModel):
    error: str

class AnalysisResponse(BaseModel):
    channel_id: Optional[str] = None
    video_analyses: Optional[List[Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None
    
# Initialize analyzers
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
youtube_analyzer = YouTubeAnalyzer(youtube_api_key=youtube_api_key)

@app.get("/")
async def root():
    return {"message": "YouTube Content Compliance Analyzer API"}

@app.post("/analyze/channel", response_model=AnalysisResponse)
async def analyze_channel(request: ChannelAnalysisRequest):
    """
    Analyze a YouTube channel for content compliance
    """
    try:
        # Initialize LLM analyzer with specified provider
        llm_analyzer = LLMAnalyzer(provider=request.llm_provider)
        
        # Get channel data (videos and transcripts)
        channel_data = youtube_analyzer.analyze_channel(request.channel_url, video_limit=request.video_limit)
        
        if not channel_data:
            raise HTTPException(status_code=404, detail="Could not extract channel data")
            
        # If no videos with transcripts were found
        if not channel_data.get('videos'):
            return {
                "channel_id": channel_data.get('channel_id', "unknown"),
                "video_analyses": [],
                "summary": {}
            }
            
        # Analyze content against compliance categories
        analysis_results = llm_analyzer.analyze_channel_content(channel_data)
        
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing channel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    Analyze a single YouTube video for content compliance
    """
    try:
        # Extract video ID from URL
        video_url = str(request.video_url).strip()
        logger.info(f"Analyzing video URL: {video_url}")
        video_id = None
        
        if "youtube.com/watch" in video_url and "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0] if "&" in video_url.split("v=")[1] else video_url.split("v=")[1]
            logger.info(f"Extracted video ID from youtube.com/watch URL: {video_id}")
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0] if "?" in video_url.split("youtu.be/")[1] else video_url.split("youtu.be/")[1]
            logger.info(f"Extracted video ID from youtu.be URL: {video_id}")
        else:
            logger.warning(f"Could not extract video ID from URL: {video_url}")
            
        if not video_id:
            raise HTTPException(status_code=400, detail=f"Invalid YouTube video URL: {video_url}")
        
        # Get video title by scraping (since we don't have the full data from API)
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(video_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text.replace(' - YouTube', '')
        logger.info(f"Retrieved video title: {title}")
        
        # Get channel ID from the video
        channel_id = youtube_analyzer.get_channel_id_from_video(video_id)
        logger.info(f"Retrieved channel ID: {channel_id}")
        
        # Get video transcript
        transcript = youtube_analyzer.get_transcript(video_id)
        if not transcript:
            logger.warning(f"No transcript available for video {video_id}")
            # Return a partial result with video info but no analysis
            return {
                "channel_id": channel_id,
                "video_analyses": [{
                    "video_id": video_id,
                    "video_title": title,
                    "video_url": video_url,
                    "analysis": {
                        "video_id": video_id,
                        "message": "No transcript available for this video. Unable to perform content analysis.",
                        "results": {}
                    }
                }],
                "summary": {}
            }
            
        logger.info(f"Retrieved transcript for video {video_id}, length: {len(transcript['full_text'])} characters")
        
        # Initialize LLM analyzer with specified provider
        llm_analyzer = LLMAnalyzer(provider=request.llm_provider)
        
        # Analyze transcript
        analysis = llm_analyzer.analyze_transcript(
            transcript_text=transcript['full_text'],
            video_title=title,
            video_id=video_id
        )
        
        logger.info(f"Analysis completed for video {video_id}, found {len(analysis.get('results', {}))} categories with violations")
        
        return {
            "channel_id": channel_id,
            "video_analyses": [{
                "video_id": video_id,
                "video_title": title,
                "video_url": video_url,
                "analysis": analysis
            }],
            "summary": {
                # Simple summary with just the results from this video
                category: {
                    "max_score": analysis.get("results", {}).get(category, {}).get("score", 0),
                    "average_score": analysis.get("results", {}).get(category, {}).get("score", 0),
                    "videos_with_violations": 1 if category in analysis.get("results", {}) else 0,
                    "total_videos": 1,
                    "examples": [{
                        "video_id": video_id,
                        "video_title": title,
                        "video_url": video_url,
                        "score": analysis.get("results", {}).get(category, {}).get("score", 0),
                        "evidence": analysis.get("results", {}).get(category, {}).get("evidence", [])[0] 
                            if analysis.get("results", {}).get(category, {}).get("evidence") else ""
                    }] if category in analysis.get("results", {}) else []
                } for category in llm_analyzer.categories_df['Category'].tolist() 
                if category in analysis.get("results", {})
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-multiple")
async def analyze_multiple_urls(request: MultipleURLsRequest):
    """
    Analyze multiple YouTube URLs (videos or channels)
    """
    try:
        # Initialize analyzers
        llm_analyzer = LLMAnalyzer(provider=request.llm_provider)
        
        # Deduplicate URLs while preserving order
        unique_urls = []
        seen = set()
        for url in request.urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        logger.info(f"Processing {len(unique_urls)} unique URLs out of {len(request.urls)} submitted")
        
        # Process each URL
        results = []
        tasks = []
        
        for url in unique_urls:
            # Normalize URL to ensure consistent handling
            url = str(url).strip()
            
            if "youtube.com/channel/" in url or "youtube.com/c/" in url or "youtube.com/@" in url:
                # Channel URL
                channel_request = ChannelAnalysisRequest(
                    channel_url=url,
                    llm_provider=request.llm_provider
                )
                tasks.append(analyze_channel(channel_request))
            elif "youtube.com/watch" in url or "youtu.be/" in url:
                # Video URL - ensure the URL is properly formatted
                video_request = VideoAnalysisRequest(
                    video_url=url,
                    llm_provider=request.llm_provider
                )
                tasks.append(analyze_video(video_request))
            else:
                logger.warning(f"Unrecognized URL format: {url}")
                results.append({
                    "url": url,
                    "error": "Unrecognized URL format"
                })
        
        # Run all tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_urls = [u for u in unique_urls if "youtube.com/" in u or "youtu.be/" in u]
        logger.info(f"Processing results for {len(valid_urls)} valid URLs")
        
        for i, (url, result) in enumerate(zip(valid_urls, task_results)):
            if isinstance(result, Exception):
                # Log the error but include it in results
                logger.error(f"Error processing URL {url}: {str(result)}")
                results.append({
                    "url": url,
                    "error": str(result)
                })
            else:
                results.append({
                    "url": url,
                    "analysis": result
                })
        
        # Create a combined summary across all successful results
        combined_summary = create_combined_summary(results)
        
        return {
            "individual_results": results,
            "combined_summary": combined_summary
        }
    except Exception as e:
        logger.error(f"Error analyzing multiple URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_combined_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a combined summary from multiple analysis results
    """
    # Initialize summary structure
    summary = {}
    total_videos = 0
    videos_with_transcripts = 0
    videos_with_errors = 0
    videos_without_transcripts = 0
    videos_with_no_violations = 0
    
    # Process each result
    for result in results:
        if "error" in result:
            videos_with_errors += 1
            continue
            
        analysis = result.get("analysis", {})
        
        # Check if this is a video without a transcript
        video_analyses = analysis.get("video_analyses", [])
        if video_analyses and "message" in video_analyses[0].get("analysis", {}) and "No transcript available" in video_analyses[0]["analysis"]["message"]:
            # Count it as a video without transcript
            total_videos += 1
            videos_without_transcripts += 1
            continue
            
        # Extract summary data
        if "summary" in analysis:
            result_summary = analysis["summary"]
            
            # Count videos
            video_count = len(analysis.get("video_analyses", []))
            total_videos += video_count
            videos_with_transcripts += video_count
            
            # Check if there are any violations
            has_violations = False
            for category, data in result_summary.items():
                if data.get("videos_with_violations", 0) > 0:
                    has_violations = True
                    break
                    
            if not has_violations:
                videos_with_no_violations += video_count
            
            # Merge summaries
            for category, data in result_summary.items():
                if category not in summary:
                    # Initialize category in combined summary
                    summary[category] = {
                        "max_score": 0.0,
                        "total_score": 0.0,
                        "videos_with_violations": 0,
                        "total_videos": 0,
                        "examples": []
                    }
                
                # Update summary data
                summary[category]["total_videos"] += video_count
                summary[category]["videos_with_violations"] += data.get("videos_with_violations", 0)
                
                # Update max score
                if data.get("max_score", 0) > summary[category]["max_score"]:
                    summary[category]["max_score"] = data["max_score"]
                
                # Add to total score for average calculation
                summary[category]["total_score"] += data.get("average_score", 0) * data.get("videos_with_violations", 0)
                
                # Add examples
                for example in data.get("examples", [])[:2]:  # Limit to top 2 examples per result
                    summary[category]["examples"].append(example)
    
    # Calculate average scores and sort examples
    for category, data in summary.items():
        # Calculate average score
        if data["videos_with_violations"] > 0:
            data["average_score"] = data["total_score"] / data["videos_with_violations"]
        else:
            data["average_score"] = 0.0
            
        # Remove temporary total_score field
        del data["total_score"]
        
        # Sort examples by score (highest first)
        data["examples"] = sorted(
            data["examples"], 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )[:5]  # Keep only top 5 examples
    
    # Add metadata about the analysis
    summary["_metadata"] = {
        "total_videos": total_videos,
        "videos_with_transcripts": videos_with_transcripts,
        "videos_without_transcripts": videos_without_transcripts,
        "videos_with_errors": videos_with_errors,
        "videos_with_no_violations": videos_with_no_violations
    }
    
    return summary

@app.get("/categories")
async def get_categories():
    """
    Get list of compliance categories with definitions
    """
    try:
        import pandas as pd
        categories_df = pd.read_csv("YouTube_Controversy_Categories.csv")
        categories = categories_df.to_dict(orient="records")
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 