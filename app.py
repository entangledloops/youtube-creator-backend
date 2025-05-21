"""
FastAPI application for YouTube content compliance analysis
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import logging
import uvicorn
import asyncio
import pandas as pd
import uuid
from datetime import datetime
import json

from youtube_analyzer import YouTubeAnalyzer
from llm_analyzer import LLMAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables...")
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger.info(f"Looking for .env file at: {env_path}")

# Force reload of environment variables
if os.path.exists(env_path):
    logger.info("Found .env file, loading variables...")
    # Force reload of environment variables
    load_dotenv(env_path, override=True)
    
    # Verify the API key was loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.error("Invalid or missing OpenAI API key in .env file")
        logger.error("Please ensure your .env file contains a valid OPENAI_API_KEY")
        raise ValueError("Invalid or missing OpenAI API key in .env file")
    
    # Log the API key that was loaded (first 4 and last 4 chars for security)
    api_key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key
    logger.info(f"Loaded OpenAI API key: {api_key_preview}")
else:
    logger.error(f"No .env file found at {env_path}")
    logger.error("Please create a .env file with your OpenAI API key")
    raise ValueError("No .env file found")

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

class CreatorAnalysisRequest(BaseModel):
    creator_url: HttpUrl
    video_limit: int = 10
    llm_provider: str = "openai"

class BulkAnalysisRequest(BaseModel):
    video_limit: int = 10
    llm_provider: str = "openai"

# Response models
class ErrorResponse(BaseModel):
    error: str

class AnalysisResponse(BaseModel):
    channel_id: str
    channel_name: str
    channel_handle: str
    video_analyses: List[dict]
    summary: dict

# Initialize analyzers
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
youtube_analyzer = YouTubeAnalyzer(youtube_api_key=youtube_api_key)

# Store for bulk analysis results
analysis_results = {}

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
                "channel_name": channel_data.get('channel_name', "Unknown"),
                "channel_handle": channel_data.get('channel_handle', "Unknown"),
                "video_analyses": [],
                "summary": {}
            }
            
        # Analyze content against compliance categories
        analysis_results = llm_analyzer.analyze_channel_content(channel_data)
        
        # Add channel name and handle to the response
        analysis_results["channel_name"] = channel_data.get('channel_name', "Unknown")
        analysis_results["channel_handle"] = channel_data.get('channel_handle', "Unknown")
        
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
                "channel_name": None,
                "channel_handle": None,
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
            "channel_name": None,
            "channel_handle": None,
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

@app.post("/api/analyze-creator", response_model=AnalysisResponse)
async def analyze_creator(request: CreatorAnalysisRequest):
    """
    Analyze a YouTube creator's recent videos for content compliance
    """
    try:
        # Initialize LLM analyzer with specified provider
        llm_analyzer = LLMAnalyzer(provider=request.llm_provider)
        
        # Convert HttpUrl to string
        creator_url = str(request.creator_url).strip()
        logger.info(f"Analyzing creator URL: {creator_url}")
        
        # Get channel data (videos and transcripts) asynchronously
        # Only use concurrent processing for OpenAI
        use_concurrent = request.llm_provider == "openai"
        channel_data = await youtube_analyzer.analyze_channel_async(
            creator_url, 
            video_limit=request.video_limit,
            use_concurrent=use_concurrent
        )
        
        if not channel_data:
            raise HTTPException(status_code=404, detail="Could not extract channel data")
            
        # If no videos with transcripts were found
        if not channel_data.get('videos'):
            return {
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_data.get('channel_name', "Unknown"),
                "channel_handle": channel_data.get('channel_handle', "Unknown"),
                "video_analyses": [],
                "summary": {}
            }
            
        # Analyze content against compliance categories using async processing
        analysis_results = await llm_analyzer.analyze_channel_content_async(channel_data)
        
        # Add channel name and handle to the response
        analysis_results["channel_name"] = channel_data.get('channel_name', "Unknown")
        analysis_results["channel_handle"] = channel_data.get('channel_handle', "Unknown")
        
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing creator: {str(e)}")
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

async def process_creator(url: str, video_limit: int, llm_provider: str, job_id: str):
    """Process a single creator and store results"""
    try:
        # Initialize LLM analyzer with specified provider
        llm_analyzer = LLMAnalyzer(provider=llm_provider)
        
        # Get channel data
        channel_data = await youtube_analyzer.analyze_channel_async(
            str(url), 
            video_limit=video_limit,
            use_concurrent=(llm_provider == "openai")
        )
        
        if not channel_data:
            analysis_results[job_id]['failed_urls'].append({
                'url': str(url),
                'error': 'Could not extract channel data'
            })
            analysis_results[job_id]['processed_urls'] += 1
            return
            
        # If no videos with transcripts were found
        if not channel_data.get('videos'):
            # Safely handle channel name and handle
            channel_name = channel_data.get('channel_name', '')
            channel_handle = channel_data.get('channel_handle', '')
            
            # Remove @ if present and ensure we have a valid string
            if isinstance(channel_name, str):
                channel_name = channel_name.replace('@', '')
            if isinstance(channel_handle, str):
                channel_handle = channel_handle.replace('@', '')
                
            # Use channel ID as fallback for name if needed
            if not channel_name and channel_data.get('channel_id'):
                channel_name = f"Channel {channel_data['channel_id']}"
            
            analysis_results[job_id]['results'][str(url)] = {
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_name or "Unknown",
                "channel_handle": channel_handle or "Unknown",
                "video_analyses": [],
                "summary": {}
            }
            analysis_results[job_id]['processed_urls'] += 1
            return
            
        # Analyze content
        analysis_result = await llm_analyzer.analyze_channel_content_async(channel_data)
        
        # Safely handle channel name and handle
        channel_name = channel_data.get('channel_name', '')
        channel_handle = channel_data.get('channel_handle', '')
        
        # Remove @ if present and ensure we have a valid string
        if isinstance(channel_name, str):
            channel_name = channel_name.replace('@', '')
        if isinstance(channel_handle, str):
            channel_handle = channel_handle.replace('@', '')
            
        # Use channel ID as fallback for name if needed
        if not channel_name and channel_data.get('channel_id'):
            channel_name = f"Channel {channel_data['channel_id']}"
        
        # Ensure we have the full channel details
        analysis_result.update({
            "channel_id": channel_data.get('channel_id', "unknown"),
            "channel_name": channel_name or "Unknown",
            "channel_handle": channel_handle or "Unknown",
            "video_analyses": analysis_result.get('video_analyses', []),
            "summary": analysis_result.get('summary', {})
        })
        
        # Store the complete result
        analysis_results[job_id]['results'][str(url)] = analysis_result
        analysis_results[job_id]['processed_urls'] += 1
        
    except Exception as e:
        logger.error(f"Error processing creator {url}: {str(e)}")
        analysis_results[job_id]['failed_urls'].append({
            'url': str(url),
            'error': str(e)
        })
        analysis_results[job_id]['processed_urls'] += 1

async def process_bulk_analysis(urls: List[str], video_limit: int, llm_provider: str, job_id: str):
    """Process multiple URLs in parallel with rate limiting"""
    try:
        # If using OpenAI, process in smaller batches to avoid rate limits
        if llm_provider == "openai":
            # Process in batches of 2 to avoid rate limits
            batch_size = 2
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                tasks = []
                for url in batch:
                    task = process_creator(url, video_limit, llm_provider, job_id)
                    tasks.append(task)
                
                # Wait for batch to complete
                await asyncio.gather(*tasks)
                
                # Add a delay between batches to avoid rate limits
                if i + batch_size < len(urls):
                    await asyncio.sleep(5)  # 5 second delay between batches
        else:
            # For non-OpenAI providers, process all at once
            tasks = []
            for url in urls:
                task = process_creator(url, video_limit, llm_provider, job_id)
                tasks.append(task)
            await asyncio.gather(*tasks)
        
        # Verify all URLs were processed
        total_processed = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
        if total_processed != len(urls):
            logger.warning(f"Job {job_id}: Processed {total_processed} URLs but expected {len(urls)}")
        
        # Mark job as complete
        analysis_results[job_id]['status'] = 'completed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['processed_urls'] = total_processed
        
    except Exception as e:
        logger.error(f"Error in bulk processing: {str(e)}")
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['error'] = str(e)
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()

@app.post("/api/bulk-analyze")
async def bulk_analyze(
    file: UploadFile = File(...),
    request: BulkAnalysisRequest = None,
    background_tasks: BackgroundTasks = None
):
    """Bulk analyze multiple YouTube creators from a CSV file.
    The file can be a single column of URLs with or without a header.
    """
    try:
        # Read CSV file with flexible header handling
        try:
            # First try with header
            df = pd.read_csv(file.file)
            # If we have a header, use the first column regardless of its name
            urls = df.iloc[:, 0].tolist()
        except:
            # If that fails, try without header
            file.file.seek(0)  # Reset file pointer
            df = pd.read_csv(file.file, header=None)
            urls = df.iloc[:, 0].tolist()
            
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            url = str(url).strip()
            if url and (url.startswith('http://') or url.startswith('https://')):
                cleaned_urls.append(url)
            else:
                logger.warning(f"Skipping invalid URL: {url}")
                
        if not cleaned_urls:
            raise HTTPException(status_code=400, detail="No valid URLs found in file")
            
        logger.info(f"Found {len(cleaned_urls)} valid URLs to process")
            
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize results storage
        analysis_results[job_id] = {
            'status': 'processing',
            'started_at': datetime.now().isoformat(),
            'total_urls': len(cleaned_urls),
            'processed_urls': 0,
            'results': {},
            'failed_urls': []
        }
        
        # Start processing in background
        background_tasks.add_task(
            process_bulk_analysis,
            cleaned_urls,
            request.video_limit if request else 10,
            request.llm_provider if request else "openai",
            job_id
        )
        
        return {
            "job_id": job_id,
            "total_urls": len(cleaned_urls),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error in bulk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bulk-analyze/{job_id}")
async def get_bulk_analysis_status(job_id: str):
    """Get status of a bulk analysis job"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    return {
        "job_id": job_id,
        "status": job['status'],
        "started_at": job['started_at'],
        "total_urls": job['total_urls'],
        "processed_urls": len(job['results']) + len(job['failed_urls']),
        "failed_urls": job['failed_urls']
    }

@app.get("/api/bulk-analyze/{job_id}/results")
async def get_bulk_analysis_results(job_id: str):
    """Get detailed results of a bulk analysis job"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
        
    return {
        "job_id": job_id,
        "status": job['status'],
        "started_at": job['started_at'],
        "completed_at": job.get('completed_at'),
        "total_urls": job['total_urls'],
        "processed_urls": job['processed_urls'],
        "results": job['results'],
        "failed_urls": job['failed_urls']
    }

@app.get("/api/bulk-analyze/{job_id}/csv")
async def download_bulk_analysis_csv(job_id: str):
    """Download bulk analysis results as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
        
    # Get all categories
    categories_df = pd.read_csv("YouTube_Controversy_Categories.csv")
    all_categories = categories_df['Category'].tolist()
        
    # Convert results to DataFrame
    rows = []
    for url, result in job['results'].items():
        row = {
            'url': url,
            'channel_id': result.get('channel_id', ''),
            'channel_name': result.get('channel_name', ''),
            'channel_handle': result.get('channel_handle', ''),
            'overall_score': 0.0  # Will be calculated
        }
        
        # Initialize all categories with 0.0
        for category in all_categories:
            row[category] = 0.0
            
        # Add category scores from results
        for category, data in result.get('summary', {}).items():
            if category in all_categories:  # Only include valid categories
                score = round(data.get('average_score', 0), 2)
                row[category] = score
                row['overall_score'] = max(row['overall_score'], score)
            
        # Round overall score
        row['overall_score'] = round(row['overall_score'], 2)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Save to temporary CSV
    filename = f"bulk_analysis_{job_id}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 