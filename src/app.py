"""
FastAPI application for YouTube content compliance analysis

Environment Variables for YouTube Rate Limiting (to prevent IP blocking):
- YOUTUBE_MAX_CONCURRENT: Maximum concurrent transcript requests (default: 5)
- YOUTUBE_REQUEST_DELAY: Delay between transcript requests in seconds (default: 0.5)
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import logging
import asyncio
import pandas as pd
import uuid
from datetime import datetime, timedelta
import json
import time

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables...")
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
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

# Get default LLM provider from environment
default_llm_provider = os.getenv("LLM_PROVIDER", "local")
logger.info(f"Default LLM provider from environment: {default_llm_provider}")

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

# Input validation models (only for active endpoints)

class CreatorAnalysisRequest(BaseModel):
    creator_url: HttpUrl
    video_limit: int = 10
    llm_provider: Optional[str] = None  # Will be set dynamically
    
    def __init__(self, **data):
        if data.get('llm_provider') is None:
            data['llm_provider'] = os.getenv("LLM_PROVIDER", "local")
        super().__init__(**data)

class BulkAnalysisRequest(BaseModel):
    video_limit: int = 10
    llm_provider: Optional[str] = None  # Will be set dynamically
    
    def __init__(self, **data):
        if data.get('llm_provider') is None:
            data['llm_provider'] = os.getenv("LLM_PROVIDER", "local")
        super().__init__(**data)

# Response models
class ErrorResponse(BaseModel):
    error: str

class AnalysisResponse(BaseModel):
    channel_id: str
    channel_name: str
    channel_handle: str
    video_analyses: List[dict]
    summary: dict
    status: str = "success"  # Add status field with default value



# Initialize analyzers
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
youtube_analyzer = YouTubeAnalyzer(youtube_api_key=youtube_api_key)

# Global storage for analysis results with detailed pipeline tracking
analysis_results = {}

# Global job queue for sequential processing
job_queue = []
current_job_id = None
job_queue_lock = asyncio.Lock()

# Global task tracking for cancellation
active_job_tasks = {}  # job_id -> list of asyncio tasks

def update_pipeline_stage(job_id: str, from_stage: str, to_stage: str, count: int = 1):
    """Helper function to update pipeline stage counters"""
    if job_id not in analysis_results:
        return
    
    stages = analysis_results[job_id]['pipeline_stages']
    
    # Decrease from_stage count
    if from_stage and from_stage in stages:
        stages[from_stage] = max(0, stages[from_stage] - count)
    
    # Increase to_stage count
    if to_stage and to_stage in stages:
        stages[to_stage] = stages[to_stage] + count

def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled"""
    if job_id not in analysis_results:
        return True
    return analysis_results[job_id]['status'] in ['cancelled', 'cancelling']

@app.get("/")
async def root():
    return {"message": "YouTube Content Compliance Analyzer API"}



@app.post("/api/analyze-creator", response_model=AnalysisResponse)
async def analyze_creator(request: CreatorAnalysisRequest):
    """
    Analyze a YouTube creator's recent videos for content compliance
    """
    try:
        # Always use environment LLM_PROVIDER (ignore frontend request)
        actual_provider = os.getenv("LLM_PROVIDER", "local")
        logger.info(f"🔧 Creator analysis using LLM provider: {actual_provider}")
        
        # Initialize LLM analyzer with environment provider (ignore request)
        llm_analyzer = LLMAnalyzer(provider=actual_provider)
        
        # Convert HttpUrl to string
        creator_url = str(request.creator_url).strip()
        logger.info(f"Analyzing creator URL: {creator_url}")
        
        # Get channel data (videos and transcripts) asynchronously
        # Only use concurrent processing for OpenAI
        use_concurrent = actual_provider == "openai"
        channel_data = await youtube_analyzer.analyze_channel_async(
            creator_url, 
            video_limit=request.video_limit,
            use_concurrent=use_concurrent
        )
        
        if not channel_data:
            raise HTTPException(status_code=404, detail="Could not extract channel data")
            
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
            
            return {
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_name or "Unknown",
                "channel_handle": channel_handle or "Unknown",
                "video_analyses": [],
                "summary": {},
                "status": "failed"
            }
        
        # Analyze content against compliance categories using async processing
        analysis_results = await llm_analyzer.analyze_channel_content_async(channel_data)
        
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
        analysis_results.update({
            "channel_id": channel_data.get('channel_id', "unknown"),
            "channel_name": channel_name or "Unknown",
            "channel_handle": channel_handle or "Unknown",
            "video_analyses": analysis_results.get('video_analyses', []),
            "summary": analysis_results.get('summary', {})
        })
        
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing creator: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

@app.get("/api/analyze-progress")
async def analyze_progress():
    """
    Server-Sent Events (SSE) endpoint for real-time progress updates
    """
    async def event_generator():
        while True:
            # This is a placeholder implementation
            # In a real implementation, you'd track actual progress from your analysis jobs
            yield {
                "event": "progress",
                "data": json.dumps({
                    "message": "Analysis in progress...",
                    "timestamp": datetime.now().isoformat()
                })
            }
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())

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

async def process_creator(url: str, video_limit: int, llm_provider: str, job_id: str):
    """Process a single creator and store results"""
    try:
        # Always use environment LLM_PROVIDER (ignore llm_provider parameter)
        actual_provider = os.getenv("LLM_PROVIDER", "local")
        llm_analyzer = LLMAnalyzer(provider=actual_provider)
        
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
                "summary": {},
                "status": "failed"
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
        
        # Store the complete result with transcript data
        final_result = {
            "url": str(url),
            "channel_id": channel_data.get('channel_id', "unknown"),
            "channel_name": channel_name or "Unknown",
            "channel_handle": channel_handle or "Unknown",
            "video_analyses": analysis_result.get('video_analyses', []),
            "summary": analysis_result.get('summary', {}),
            "original_videos": channel_data.get('videos', [])  # Store original video data with transcripts
        }
        
        analysis_results[job_id]['results'][str(url)] = final_result
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
        
        # Create channel summaries from individual video results
        await create_channel_summaries(job_id)
        
        # Clean up task tracking
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]
        
    except Exception as e:
        logger.error(f"Error in bulk processing: {str(e)}")
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['error'] = str(e)
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        
        # Clean up task tracking on failure
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]

@app.post("/api/bulk-analyze")
async def bulk_analyze(
    file: UploadFile = File(...),
    request: BulkAnalysisRequest = None
):
    """Bulk analyze multiple YouTube creators from a CSV file with job queuing.
    The file can be a single column of URLs with or without a header.
    Jobs are processed sequentially to avoid resource conflicts.
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
        
        # Always use environment LLM_PROVIDER (ignore frontend request)
        llm_provider = os.getenv("LLM_PROVIDER", "local")
        video_limit = request.video_limit if request else 10
        
        # Initialize results storage with detailed pipeline tracking
        analysis_results[job_id] = {
            'status': 'queued',
            'started_at': datetime.now().isoformat(),
            'total_urls': len(cleaned_urls),
            'processed_urls': 0,
            'results': {},
            'failed_urls': [],
            'original_videos': {},  # Store video data for evidence API
            
            # Add video-level tracking for better progress reporting
            'video_progress': {
                'total_videos_discovered': 0,
                'videos_with_transcripts': 0,
                'videos_analyzed_by_llm': 0,
                'videos_completed': 0
            },
            
            # Detailed pipeline stage tracking
            'pipeline_stages': {
                'queued_for_transcripts': len(cleaned_urls),  # URLs waiting to start transcript fetch
                'fetching_transcripts': 0,                   # URLs currently pulling transcripts
                'queued_for_llm': 0,                        # URLs waiting for LLM analysis
                'llm_processing': 0,                        # URLs currently being analyzed by LLM
                'queued_for_results': 0,                    # URLs waiting for result processing
                'result_processing': 0,                     # URLs being processed into final results
                'completed': 0,                             # Successfully completed URLs
                'failed': 0                                 # Failed URLs
            },
            
            # Performance tracking for ETA calculation
            'performance_stats': {
                'transcript_times': [],
                'llm_times': [],
                'overall_start_time': time.time(),
                'urls_completed_timestamps': [],
                'estimated_completion_time': None,
                'processing_rate_per_minute': 0.0
            },
            
            # Job queue information
            'queue_position': 0,  # Will be updated when queued
            'estimated_start_time': None,
            'estimated_wait_minutes': 0
        }
        
        # Handle job queuing
        async with job_queue_lock:
            global current_job_id
            
            if current_job_id is None:
                # No job running, start immediately
                current_job_id = job_id
                analysis_results[job_id]['status'] = 'processing'
                analysis_results[job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting job {job_id} immediately (no queue)")
                
                # Start processing
                task = asyncio.create_task(
                    process_job_with_cleanup(job_id, cleaned_urls, video_limit, llm_provider)
                )
                
                # Track task for cancellation
                active_job_tasks[job_id] = [task]
            else:
                # Job is running, add to queue
                job_queue.append({
                    'job_id': job_id,
                    'urls': cleaned_urls,
                    'video_limit': video_limit,
                    'llm_provider': llm_provider
                })
                
                # Update queue positions for all queued jobs
                for i, queued_job in enumerate(job_queue):
                    queued_job_id = queued_job['job_id']
                    analysis_results[queued_job_id]['queue_position'] = i + 1
                    
                    # Estimate wait time based on current job progress
                    if current_job_id and current_job_id in analysis_results:
                        current_job = analysis_results[current_job_id]
                        current_progress = current_job['processed_urls'] / current_job['total_urls'] if current_job['total_urls'] > 0 else 0
                        current_remaining_urls = current_job['total_urls'] - current_job['processed_urls']
                        
                        # Estimate time per URL based on current job performance
                        elapsed_time = time.time() - current_job['performance_stats']['overall_start_time']
                        if current_job['processed_urls'] > 0:
                            time_per_url = elapsed_time / current_job['processed_urls']
                        else:
                            time_per_url = 30  # Default estimate: 30 seconds per URL
                        
                        # Calculate wait time
                        current_job_remaining_minutes = (current_remaining_urls * time_per_url) / 60
                        
                        # Add time for jobs ahead in queue
                        jobs_ahead_time = 0
                        for j in range(i):
                            ahead_job = job_queue[j]
                            jobs_ahead_time += (len(ahead_job['urls']) * time_per_url) / 60
                        
                        total_wait_minutes = current_job_remaining_minutes + jobs_ahead_time
                        analysis_results[queued_job_id]['estimated_wait_minutes'] = max(1, int(total_wait_minutes))
                        analysis_results[queued_job_id]['estimated_start_time'] = (
                            datetime.now() + timedelta(minutes=total_wait_minutes)
                        ).isoformat()
                
                logger.info(f"📋 Queued job {job_id} at position {analysis_results[job_id]['queue_position']} (estimated wait: {analysis_results[job_id]['estimated_wait_minutes']} minutes)")
        
        # Small delay to ensure job is properly stored before returning
        await asyncio.sleep(0.1)
        
        # Return response with queue information
        response = {
            "job_id": job_id,
            "total_urls": len(cleaned_urls),
            "status": analysis_results[job_id]['status']
        }
        
        # Add queue information if job is queued
        if analysis_results[job_id]['status'] == 'queued':
            response.update({
                "queue_position": analysis_results[job_id]['queue_position'],
                "estimated_wait_minutes": analysis_results[job_id]['estimated_wait_minutes'],
                "estimated_start_time": analysis_results[job_id]['estimated_start_time'],
                "message": f"Your job is queued at position {analysis_results[job_id]['queue_position']}. Estimated wait time: {analysis_results[job_id]['estimated_wait_minutes']} minutes."
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Error in bulk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_job_with_cleanup(job_id: str, urls: List[str], video_limit: int, llm_provider: str):
    """Process a job and handle queue cleanup when done"""
    try:
        await process_creators_pipeline(job_id, urls, video_limit, llm_provider)
    finally:
        # Clean up and start next job
        async with job_queue_lock:
            global current_job_id
            current_job_id = None
            
            # Start next job if any are queued
            if job_queue:
                next_job = job_queue.pop(0)
                current_job_id = next_job['job_id']
                
                # Update status and start processing
                analysis_results[current_job_id]['status'] = 'processing'
                analysis_results[current_job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting next queued job {current_job_id}")
                
                task = asyncio.create_task(
                    process_job_with_cleanup(
                        current_job_id,
                        next_job['urls'],
                        next_job['video_limit'],
                        next_job['llm_provider']
                    )
                )
                
                # Track task for cancellation
                active_job_tasks[current_job_id] = [task]

@app.get("/api/bulk-analyze/{job_id}/status")
async def get_bulk_analysis_status_alias(job_id: str):
    """Alias for get_bulk_analysis_status to match frontend expectations"""
    logger.info(f"Status alias request for job_id: {job_id}")
    return await get_bulk_analysis_status(job_id)

async def get_bulk_analysis_status(job_id: str):
    """Get detailed status of a bulk analysis job with pipeline breakdown and ETA"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Calculate counts for status
    successful_count = len(job['results'])
    failed_count = len(job['failed_urls'])
    total_processed = successful_count + failed_count
    
    # Get video-level progress for more accurate status
    video_progress = job.get('video_progress', {})
    total_videos = video_progress.get('total_videos_discovered', 0)
    videos_completed = video_progress.get('videos_completed', 0)
    videos_with_transcripts = video_progress.get('videos_with_transcripts', 0)
    videos_analyzed = video_progress.get('videos_analyzed_by_llm', 0)
    
    # DEBUG: Log what we're seeing
    logger.info(f"🔍 STATUS REQUEST DEBUG for job {job_id}:")
    logger.info(f"   📊 Job status: {job['status']}")
    logger.info(f"   📊 Successful channels: {successful_count}")
    logger.info(f"   📊 Failed URLs: {failed_count}")
    logger.info(f"   📊 Total processed: {total_processed}")
    logger.info(f"   📊 Total URLs: {job['total_urls']}")
    logger.info(f"   📊 Video Progress: {total_videos} discovered, {videos_with_transcripts} transcripts, {videos_analyzed} analyzed, {videos_completed} completed")
    logger.info(f"   📊 Pipeline stages: {job.get('pipeline_stages', {})}")
    
    # Calculate progress based on video completion if we have video data
    if total_videos > 0:
        video_progress_percentage = (videos_completed / total_videos) * 100
        effective_processed = videos_completed
        effective_total = total_videos
    else:
        # Fallback to channel-level progress
        video_progress_percentage = (total_processed / job['total_urls'] * 100) if job['total_urls'] > 0 else 0
        effective_processed = total_processed
        effective_total = job['total_urls']
    
    # Calculate ETA if job is processing
    eta_info = {}
    if job['status'] == 'processing':
        elapsed_time = time.time() - job['performance_stats']['overall_start_time']
        
        if effective_processed > 0:
            # Calculate processing rate using effective progress
            processing_rate_per_second = effective_processed / elapsed_time
            processing_rate_per_minute = processing_rate_per_second * 60
            
            # Estimate remaining time
            remaining_items = effective_total - effective_processed
            estimated_seconds_remaining = remaining_items / processing_rate_per_second if processing_rate_per_second > 0 else 0
            estimated_minutes_remaining = estimated_seconds_remaining / 60
            
            # Update job performance stats
            job['performance_stats']['processing_rate_per_minute'] = processing_rate_per_minute
            job['performance_stats']['estimated_completion_time'] = (
                datetime.now() + timedelta(seconds=estimated_seconds_remaining)
            ).isoformat()
            
            eta_info = {
                "estimated_completion_time": job['performance_stats']['estimated_completion_time'],
                "estimated_minutes_remaining": max(1, int(estimated_minutes_remaining)),
                "processing_rate_per_minute": round(processing_rate_per_minute, 2),
                "elapsed_minutes": round(elapsed_time / 60, 1),
                "progress_type": "videos" if total_videos > 0 else "channels"
            }
            
            logger.info(f"   📊 ETA info: {eta_info}")
        else:
            eta_info = {
                "estimated_completion_time": None,
                "estimated_minutes_remaining": None,
                "processing_rate_per_minute": 0,
                "elapsed_minutes": round(elapsed_time / 60, 1),
                "progress_type": "videos" if total_videos > 0 else "channels"
            }
            logger.info(f"   📊 ETA info (no progress yet): {eta_info}")
    
    # Build detailed response
    response = {
        "job_id": job_id,
        "status": job['status'],
        "started_at": job['started_at'],
        "completed_at": job.get('completed_at'),
        "total_urls": job['total_urls'],
        "processed_urls": job['processed_urls'],
        
        # Basic progress info (for backward compatibility)
        "progress": {
            "successful": successful_count,
            "failed": failed_count,
            "total_processed": total_processed,
            "percentage": video_progress_percentage
        },
        
        # Video-level progress details
        "video_progress": {
            "total_videos_discovered": total_videos,
            "videos_with_transcripts": videos_with_transcripts,
            "videos_analyzed_by_llm": videos_analyzed,
            "videos_completed": videos_completed,
            "video_percentage": video_progress_percentage
        },
        
        # Detailed pipeline stage breakdown
        "pipeline_stages": job.get('pipeline_stages', {}),
        
        # ETA and performance information
        "eta": eta_info,
        
        # Queue information (if applicable)
        "queue_info": {}
    }
    
    # Add queue information for queued jobs
    if job['status'] == 'queued':
        response["queue_info"] = {
            "queue_position": job.get('queue_position', 0),
            "estimated_wait_minutes": job.get('estimated_wait_minutes', 0),
            "estimated_start_time": job.get('estimated_start_time'),
            "message": f"Your job is queued at position {job.get('queue_position', 0)}. Estimated wait time: {job.get('estimated_wait_minutes', 0)} minutes."
        }
    
    # DEBUG: Log what we're sending to frontend
    logger.info(f"   📤 Sending to frontend - Progress: {response['progress']['percentage']:.1f}%")
    logger.info(f"   📤 Pipeline stages being sent: {response['pipeline_stages']}")
    
    return response

@app.get("/api/bulk-analyze/{job_id}/results")
async def get_bulk_analysis_results(job_id: str):
    """Get detailed results of a bulk analysis job"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Allow access to partial results for cancelled jobs
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
    
    # Calculate counts for debugging
    successful_count = len(job['results'])
    failed_count = len(job['failed_urls'])
    total_processed = successful_count + failed_count
    
    # Debug logging to help frontend troubleshooting
    logger.info(f"🔍 RESULTS ENDPOINT DEBUG for job {job_id}:")
    logger.info(f"   📊 Job status: {job['status']}")
    logger.info(f"   📊 Successful results: {successful_count}")
    logger.info(f"   📊 Failed URLs: {failed_count}")
    logger.info(f"   📊 Total processed: {total_processed}")
    logger.info(f"   📊 Original total_urls: {job['total_urls']}")
    
    if failed_count > 0:
        logger.info(f"   ❌ Failed URLs list: {[f['url'] for f in job['failed_urls']]}")
        logger.info(f"   ❌ Failed error types: {[f.get('error_type', 'unknown') for f in job['failed_urls']]}")
        for i, failed_url in enumerate(job['failed_urls'], 1):
            logger.info(f"   ❌ Failure {i}: {failed_url['url']}")
            logger.info(f"      └─ Error: {failed_url['error']}")
            if 'channel_name' in failed_url:
                logger.info(f"      └─ Channel: {failed_url['channel_name']} ({failed_url.get('video_count', 0)} videos)")
        
    response_data = {
        "job_id": job_id,
        "status": job['status'],
        "started_at": job['started_at'],
        "completed_at": job.get('completed_at'),
        "total_urls": job['total_urls'],
        "processed_urls": job['processed_urls'],
        "results": job['results'],
        "failed_urls": job['failed_urls'],
        # Add explicit counts for frontend debugging
        "summary_counts": {
            "successful": successful_count,
            "failed": failed_count, 
            "total_processed": total_processed
        }
    }
    
    # Add cancellation info if applicable
    if job['status'] == 'cancelled':
        response_data["cancellation_info"] = {
            "partial_results": True,
            "message": f"Job was cancelled. Partial results available for {successful_count} channels."
        }
    
    logger.info(f"   📤 Sending to frontend: {successful_count} results, {failed_count} failed_urls")
    return response_data

@app.delete("/api/bulk-analyze/{job_id}")
async def cancel_bulk_analysis(job_id: str):
    """Cancel an ongoing bulk analysis job gracefully"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Can only cancel jobs that are processing or queued
    if job['status'] not in ['processing', 'queued']:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job['status']}")
    
    logger.info(f"🛑 CANCELLATION REQUEST for job {job_id} (status: {job['status']})")
    
    async with job_queue_lock:
        global current_job_id
        
        if job['status'] == 'queued':
            # Remove from queue
            global job_queue
            job_queue = [j for j in job_queue if j['job_id'] != job_id]
            
            # Update queue positions for remaining jobs
            for i, queued_job in enumerate(job_queue):
                queued_job_id = queued_job['job_id']
                analysis_results[queued_job_id]['queue_position'] = i + 1
            
            # Mark as cancelled
            analysis_results[job_id]['status'] = 'cancelled'
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"✅ Cancelled queued job {job_id}")
            
        elif job['status'] == 'processing' and current_job_id == job_id:
            # Cancel active job
            analysis_results[job_id]['status'] = 'cancelling'
            
            # Cancel all active tasks for this job
            if job_id in active_job_tasks:
                logger.info(f"🛑 Cancelling {len(active_job_tasks[job_id])} active tasks for job {job_id}")
                for task in active_job_tasks[job_id]:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to finish cancelling
                try:
                    await asyncio.gather(*active_job_tasks[job_id], return_exceptions=True)
                except Exception as e:
                    logger.warning(f"Error during task cancellation: {str(e)}")
                
                # Clean up task tracking
                del active_job_tasks[job_id]
            
            # Mark as cancelled and preserve partial results
            analysis_results[job_id]['status'] = 'cancelled'
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
            
            # Reset current job and start next in queue
            current_job_id = None
            
            # Start next job if any are queued
            if job_queue:
                next_job = job_queue.pop(0)
                current_job_id = next_job['job_id']
                
                # Update status and start processing
                analysis_results[current_job_id]['status'] = 'processing'
                analysis_results[current_job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting next queued job {current_job_id}")
                
                task = asyncio.create_task(
                    process_job_with_cleanup(
                        current_job_id,
                        next_job['urls'],
                        next_job['video_limit'],
                        next_job['llm_provider']
                    )
                )
                
                # Track task for cancellation
                active_job_tasks[current_job_id] = [task]
            
            logger.info(f"✅ Cancelled active job {job_id}")
    
    # Calculate final statistics for the cancelled job
    successful_count = len(job['results'])
    failed_count = len(job['failed_urls'])
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled successfully",
        "partial_results": {
            "successful_channels": successful_count,
            "failed_urls": failed_count,
            "total_processed": successful_count + failed_count,
            "results_preserved": successful_count > 0
        },
        "cancelled_at": analysis_results[job_id]['completed_at']
    }

@app.get("/api/bulk-analyze/{job_id}/csv")
async def download_bulk_analysis_csv(job_id: str):
    """Download bulk analysis results as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    
    # Allow CSV download for cancelled jobs (partial results)
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
        
    # Get all categories - fix the path to be relative to project root
    src_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(src_dir)
    categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
    categories_df = pd.read_csv(categories_file)
    all_categories = categories_df['Category'].tolist()
    
    # Check if we have any results
    if not job['results']:
        raise HTTPException(status_code=404, detail="No results available for download")
        
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
    
    # Add metadata for cancelled jobs
    filename_suffix = "_partial" if job['status'] == 'cancelled' else ""
    filename = f"bulk_analysis_{job_id}{filename_suffix}.csv"
    df.to_csv(filename, index=False)
    
    return FileResponse(
        filename,
        media_type='text/csv',
        filename=filename
    )

@app.get("/api/bulk-analyze/{job_id}/evidence")
async def download_bulk_analysis_evidence(job_id: str):
    """Download detailed evidence and transcripts as JSON"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
    
    # Build structured evidence data
    evidence_data = {
        "job_id": job_id,
        "analysis_date": job.get('started_at'),
        "completed_date": job.get('completed_at'),
        "total_creators": len(job['results']),
        "creators": []
    }
    
    for creator_url, result in job['results'].items():
        creator_data = {
            "creator_url": creator_url,
            "channel_name": result.get('channel_name', ''),
            "channel_handle": result.get('channel_handle', ''),
            "channel_id": result.get('channel_id', ''),
            "videos": []
        }
        
        for video_analysis in result.get('video_analyses', []):
            video_data = {
                "video_url": video_analysis.get('video_url', ''),
                "video_title": video_analysis.get('video_title', ''),
                "video_id": video_analysis.get('video_id', ''),
                "transcript": "Transcript not available",  # Will be populated below
                "violations": []
            }
            
            # Get transcript from original videos data
            video_id = video_analysis.get('video_id', '')
            for original_video in result.get('original_videos', []):
                if original_video.get('id') == video_id and original_video.get('transcript'):
                    video_data["transcript"] = original_video['transcript']['full_text']
                    break
            
            # Get analysis data
            analysis = video_analysis.get('analysis', {})
            
            # Add violations with evidence
            for category, violation_data in analysis.get('results', {}).items():
                violation = {
                    "category": category,
                    "score": violation_data.get('score', 0),
                    "justification": violation_data.get('justification', ''),
                    "evidence": violation_data.get('evidence', [])
                }
                video_data["violations"].append(violation)
            
            creator_data["videos"].append(video_data)
        
        evidence_data["creators"].append(creator_data)
    
    # Return as streaming JSON response
    json_str = json.dumps(evidence_data, indent=2, ensure_ascii=False)
    
    def generate():
        yield json_str
    
    return StreamingResponse(
        generate(), 
        media_type='application/json',
        headers={
            "Content-Disposition": f"attachment; filename=evidence_{job_id}.json"
        }
    )

async def process_creators_pipeline(job_id: str, urls: List[str], video_limit: int, llm_provider: str):
    """Queue-based pipeline with proper video-level rate limiting"""
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting pipeline for job {job_id} with {len(urls)} URLs")
        logger.info(f"⚖️ Using 8 channel discovery workers, 5 transcript workers, 10 LLM workers")
        
        # Always use environment LLM_PROVIDER (ignore any frontend input)
        actual_llm_provider = os.getenv("LLM_PROVIDER", "local")
        youtube_analyzer = YouTubeAnalyzer()
        logger.info(f"Creating LLMAnalyzer with provider='{actual_llm_provider}' in pipeline")
        llm_analyzer = LLMAnalyzer(provider=actual_llm_provider)
        
        # Create queues for pipeline stages
        channel_queue = asyncio.Queue(maxsize=1000)  # Channel URLs to discover videos
        video_queue = asyncio.Queue(maxsize=1000)    # Individual videos for transcript fetching
        llm_queue = asyncio.Queue(maxsize=1000)      # Videos with transcripts for LLM analysis
        result_queue = asyncio.Queue(maxsize=1000)   # LLM results for final processing
        
        # Enhanced timing and queue tracking
        timing_stats = {
            'channel_discovery_times': [],
            'transcript_times': [],
            'llm_times': [],
            'total_times': [],
            'queue_depths': {
                'channel': [],
                'video': [],
                'llm': [],
                'result': []
            },
            'timestamps': []
        }
        
        # Start all channel URLs for discovery
        logger.info(f"📋 Queuing all {len(urls)} channel URLs for video discovery")
        for url in urls:
            await channel_queue.put({
                'url': url,
                'video_limit': video_limit,
                'start_time': time.time()
            })
        
        # Capture initial queue state
        initial_channel_size = channel_queue.qsize()
        logger.info(f"📊 Initial queue state: Channels={initial_channel_size}, Videos=0, LLM=0, Result=0")
        
        # Record initial queue depths
        timing_stats['queue_depths']['channel'].append(initial_channel_size)
        timing_stats['queue_depths']['video'].append(0)
        timing_stats['queue_depths']['llm'].append(0)
        timing_stats['queue_depths']['result'].append(0)
        timing_stats['timestamps'].append(time.time())
        
        # Start workers
        channel_workers = []
        video_workers = []
        llm_workers = []
        result_workers = []
        
        # 8 channel discovery workers (fast, just fetching video lists)
        for i in range(8):
            worker = asyncio.create_task(
                channel_discovery_worker(i, channel_queue, video_queue, youtube_analyzer, timing_stats, job_id)
            )
            channel_workers.append(worker)
        
        # 5 video transcript workers (rate-limited, fetching individual video transcripts)
        for i in range(5):
            worker = asyncio.create_task(
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id)
            )
            video_workers.append(worker)
        
        # 10 LLM workers 
        for i in range(10):
            worker = asyncio.create_task(
                llm_worker(i, llm_queue, result_queue, llm_analyzer, timing_stats, job_id)
            )
            llm_workers.append(worker)
        
        # 5 result workers
        for i in range(5):
            worker = asyncio.create_task(
                result_worker(i, result_queue, timing_stats, job_id)
            )
            result_workers.append(worker)
        
        # Track all worker tasks for cancellation
        all_workers = channel_workers + video_workers + llm_workers + result_workers
        if job_id in active_job_tasks:
            active_job_tasks[job_id].extend(all_workers)
        else:
            active_job_tasks[job_id] = all_workers
        
        # Monitor progress and queue depths with enhanced tracking
        monitor_task = asyncio.create_task(
            monitor_pipeline_detailed(job_id, channel_queue, video_queue, llm_queue, result_queue, len(urls), timing_stats)
        )
        
        # Wait for all work to complete
        logger.info("⏳ Waiting for all channel discovery to complete...")
        await channel_queue.join()
        
        logger.info("⏳ Waiting for all video transcript work to complete...")
        await video_queue.join()
        
        logger.info("⏳ Waiting for all LLM work to complete...")
        await llm_queue.join()
        
        logger.info("⏳ Waiting for all result work to complete...")
        await result_queue.join()
        
        # Cancel workers and monitor
        for workers in [channel_workers, video_workers, llm_workers, result_workers]:
            for worker in workers:
                worker.cancel()
        monitor_task.cancel()
        
        # Wait a moment for final monitoring data
        await asyncio.sleep(0.5)
        
        # Calculate comprehensive final statistics
        total_time = time.time() - start_time
        successful = len(analysis_results[job_id]['results'])
        failed = len(analysis_results[job_id]['failed_urls'])
        total_processed = successful + failed
        
        # Log comprehensive final statistics
        logger.info("=" * 100)
        logger.info(f"🏁 FINAL STATISTICS for job {job_id}")
        logger.info("=" * 100)
        logger.info(f"📊 PROCESSING SUMMARY:")
        logger.info(f"   └─ URLs submitted: {len(urls)}")
        logger.info(f"   └─ URLs processed: {total_processed}")
        logger.info(f"   └─ ✅ Successful: {successful}")
        logger.info(f"   └─ ❌ Failed: {failed}")
        logger.info(f"   └─ ⏱️  Total time: {total_time:.2f}s")
        logger.info(f"   └─ 🚀 Throughput: {total_processed/total_time:.2f} URLs/sec")
        
        # Timing analysis
        if timing_stats['channel_discovery_times']:
            discovery_times = sorted(timing_stats['channel_discovery_times'])
            discovery_p50 = discovery_times[len(discovery_times)//2]
            discovery_p95 = discovery_times[int(len(discovery_times)*0.95)]
            logger.info(f"📈 CHANNEL DISCOVERY TIMING:")
            logger.info(f"   └─ Count: {len(discovery_times)} operations")
            logger.info(f"   └─ P50: {discovery_p50:.2f}s | P95: {discovery_p95:.2f}s")
            logger.info(f"   └─ Min: {min(discovery_times):.2f}s | Max: {max(discovery_times):.2f}s")
        
        if timing_stats['transcript_times']:
            transcript_times = sorted(timing_stats['transcript_times'])
            transcript_p50 = transcript_times[len(transcript_times)//2]
            transcript_p95 = transcript_times[int(len(transcript_times)*0.95)]
            logger.info(f"📈 VIDEO TRANSCRIPT TIMING:")
            logger.info(f"   └─ Count: {len(transcript_times)} operations")
            logger.info(f"   └─ P50: {transcript_p50:.2f}s | P95: {transcript_p95:.2f}s")
            logger.info(f"   └─ Min: {min(transcript_times):.2f}s | Max: {max(transcript_times):.2f}s")
        
        if timing_stats['llm_times']:
            llm_times = sorted(timing_stats['llm_times'])
            llm_p50 = llm_times[len(llm_times)//2]
            llm_p95 = llm_times[int(len(llm_times)*0.95)]
            logger.info(f"🤖 LLM ANALYSIS TIMING:")
            logger.info(f"   └─ Count: {len(llm_times)} operations")
            logger.info(f"   └─ P50: {llm_p50:.2f}s | P95: {llm_p95:.2f}s")
            logger.info(f"   └─ Min: {min(llm_times):.2f}s | Max: {max(llm_times):.2f}s")
        
        # Queue depth analysis for bottleneck identification
        if timing_stats['queue_depths']['channel']:
            channel_depths = timing_stats['queue_depths']['channel']
            video_depths = timing_stats['queue_depths']['video']
            llm_depths = timing_stats['queue_depths']['llm']
            result_depths = timing_stats['queue_depths']['result']
            
            logger.info(f"📊 QUEUE DEPTH ANALYSIS:")
            logger.info(f"   📋 Channel Queue:")
            logger.info(f"      └─ Avg: {sum(channel_depths)/len(channel_depths):.1f} | Max: {max(channel_depths)} | Min: {min(channel_depths)}")
            logger.info(f"   🎬 Video Queue:")
            logger.info(f"      └─ Avg: {sum(video_depths)/len(video_depths):.1f} | Max: {max(video_depths)} | Min: {min(video_depths)}")
            logger.info(f"   🤖 LLM Queue:")
            logger.info(f"      └─ Avg: {sum(llm_depths)/len(llm_depths):.1f} | Max: {max(llm_depths)} | Min: {min(llm_depths)}")
            logger.info(f"   📝 Result Queue:")
            logger.info(f"      └─ Avg: {sum(result_depths)/len(result_depths):.1f} | Max: {max(result_depths)} | Min: {min(result_depths)}")
            
            # Bottleneck analysis
            avg_channel = sum(channel_depths)/len(channel_depths) if channel_depths else 0
            avg_video = sum(video_depths)/len(video_depths) if video_depths else 0
            avg_llm = sum(llm_depths)/len(llm_depths) if llm_depths else 0
            avg_result = sum(result_depths)/len(result_depths) if result_depths else 0
            
            logger.info(f"🔍 BOTTLENECK ANALYSIS:")
            max_avg = max(avg_channel, avg_video, avg_llm, avg_result)
            if avg_video > max(avg_channel, avg_llm, avg_result) * 1.5:
                logger.info(f"   └─ 🚨 VIDEO TRANSCRIPT stage is the bottleneck (avg queue: {avg_video:.1f})")
                logger.info(f"      └─ Consider adding more video transcript workers or reducing rate limiting")
            elif avg_llm > max(avg_channel, avg_video, avg_result) * 1.5:
                logger.info(f"   └─ 🚨 LLM stage is the bottleneck (avg queue: {avg_llm:.1f})")
                logger.info(f"      └─ Consider adding more LLM workers or optimizing prompts")
            elif avg_result > max(avg_channel, avg_video, avg_llm) * 1.5:
                logger.info(f"   └─ 🚨 RESULT stage is the bottleneck (avg queue: {avg_result:.1f})")
                logger.info(f"      └─ Consider adding more result workers")
            else:
                logger.info(f"   └─ ✅ No significant bottlenecks detected - balanced pipeline")
        
        # Error analysis
        if failed > 0:
            error_types = {}
            for failed_url in analysis_results[job_id]['failed_urls']:
                error_type = failed_url.get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            logger.info(f"❌ ERROR ANALYSIS:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   └─ {error_type}: {count} failures")
        
        # Log any discrepancies
        if total_processed != len(urls):
            logger.error(f"🚨 DISCREPANCY DETECTED:")
            logger.error(f"   └─ Expected: {len(urls)} URLs")
            logger.error(f"   └─ Processed: {total_processed} URLs")
            logger.error(f"   └─ Missing: {len(urls) - total_processed} URLs")
        else:
            logger.info(f"✅ PROCESSING COMPLETE: All {len(urls)} URLs accounted for")
        
        logger.info("=" * 100)
        
        # Mark job as complete
        analysis_results[job_id]['status'] = 'completed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['processed_urls'] = total_processed
        
        # Create channel summaries from individual video results
        await create_channel_summaries(job_id)
        
        # Clean up task tracking
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]
        
    except Exception as e:
        logger.error(f"💥 Error in pipeline processing: {str(e)}")
        if job_id in analysis_results:
            analysis_results[job_id]['status'] = 'failed'
            analysis_results[job_id]['error'] = str(e)
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
            
            # Clean up task tracking on failure
            if job_id in active_job_tasks:
                del active_job_tasks[job_id]

async def create_channel_summaries(job_id: str):
    """Create channel summaries from individual video results"""
    try:
        logger.info(f"📊 Creating channel summaries for job {job_id}")
        
        # Get categories for summary
        src_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(src_dir)
        categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
        categories_df = pd.read_csv(categories_file)
        all_categories = categories_df['Category'].tolist()
        
        # Process each channel's video results
        for url, channel_data in analysis_results[job_id]['results'].items():
            if not channel_data.get('video_analyses'):
                continue
                
            # Create summary across all videos for this channel
            summary = {}
            for category in all_categories:
                category_violations = []
                max_score = 0
                total_score = 0
                videos_with_violations = 0
                
                for video_analysis in channel_data.get('video_analyses', []):
                    if category in video_analysis.get("analysis", {}).get("results", {}):
                        violation = video_analysis["analysis"]["results"][category]
                        score = violation.get("score", 0)
                        
                        if score > 0:
                            videos_with_violations += 1
                            max_score = max(max_score, score)
                            total_score += score
                            
                            category_violations.append({
                                "video_id": video_analysis["video_id"],
                                "video_title": video_analysis["video_title"],
                                "video_url": video_analysis["video_url"],
                                "score": score,
                                "evidence": violation.get("evidence", [])[0] if violation.get("evidence") else ""
                            })
                
                if videos_with_violations > 0:
                    summary[category] = {
                        "max_score": max_score,
                        "average_score": total_score / videos_with_violations,
                        "videos_with_violations": videos_with_violations,
                        "total_videos": len(channel_data.get('video_analyses', [])),
                        "examples": sorted(category_violations, key=lambda x: x["score"], reverse=True)[:5]
                    }
            
            # Update the channel data with summary
            analysis_results[job_id]['results'][url]['summary'] = summary
            
        # Update final processed count (count channels, not videos)
        analysis_results[job_id]['processed_urls'] = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
        
        logger.info(f"✅ Created summaries for {len(analysis_results[job_id]['results'])} channels")
        
    except Exception as e:
        logger.error(f"💥 Error creating channel summaries: {str(e)}")

async def channel_discovery_worker(worker_id: int, channel_queue: asyncio.Queue, video_queue: asyncio.Queue, 
                                  youtube_analyzer, timing_stats: dict, job_id: str):
    """Worker that discovers videos in channels and feeds individual videos to video queue"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id):
                logger.info(f"📋 Channel Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await channel_queue.get()
            url = item['url']
            video_limit = item['video_limit']
            start_time = time.time()
            
            # Update pipeline stage: queued -> fetching
            update_pipeline_stage(job_id, 'queued_for_transcripts', 'fetching_transcripts')
            
            logger.info(f"📋 Channel Worker {worker_id}: Discovering videos for {url}")
            
            # Get channel info and video list (without transcripts yet)
            channel_id, channel_name, channel_handle = youtube_analyzer.extract_channel_info_from_url(url)
            
            if not channel_id:
                error_msg = f"Failed to access YouTube channel. The URL may be invalid, private, or the channel may not exist: {url}"
                logger.warning(f"❌ Channel Worker {worker_id}: Invalid channel URL - {url}")
                
                # Update pipeline stage: fetching -> failed
                update_pipeline_stage(job_id, 'fetching_transcripts', 'failed')
                
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': error_msg,
                    'error_type': 'invalid_channel'
                })
            else:
                # Get video list from channel
                videos = youtube_analyzer.get_videos_from_channel(channel_id, limit=video_limit)
                
                if not videos:
                    error_msg = f"Channel '{channel_name or 'Unknown'}' has no videos available for analysis"
                    logger.warning(f"❌ Channel Worker {worker_id}: No videos found for {url}")
                    
                    # Update pipeline stage: fetching -> failed
                    update_pipeline_stage(job_id, 'fetching_transcripts', 'failed')
                    
                    analysis_results[job_id]['failed_urls'].append({
                        'url': url,
                        'error': error_msg,
                        'error_type': 'no_videos',
                        'channel_name': channel_name or 'Unknown'
                    })
                else:
                    discovery_time = time.time() - start_time
                    timing_stats['channel_discovery_times'].append(discovery_time)
                    
                    logger.info(f"✅ Channel Worker {worker_id}: Found {len(videos)} videos for {url} in {discovery_time:.2f}s")
                    
                    # Add each video to video queue for transcript processing
                    for video in videos:
                        # Update pipeline stage: fetching -> queued for transcripts
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_llm')
                        
                        # Update video discovery count
                        analysis_results[job_id]['video_progress']['total_videos_discovered'] += 1
                        
                        await video_queue.put({
                            'url': url,  # Original channel URL
                            'video_id': video['id'],
                            'video_title': video['title'],
                            'video_url': video['url'],
                            'channel_id': channel_id,
                            'channel_name': channel_name,
                            'channel_handle': channel_handle,
                            'start_time': time.time()
                        })
            
            channel_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown') if 'item' in locals() else 'unknown'
            error_msg = f"Failed to retrieve channel data. There was a technical error accessing YouTube for this URL: {str(e)}"
            
            logger.error(f"💥 Channel discovery worker {worker_id} error processing {url}: {e}")
            
            # Update pipeline stage: fetching -> failed
            update_pipeline_stage(job_id, 'fetching_transcripts', 'failed')
            
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'channel_discovery_error'
            })
            channel_queue.task_done()

async def video_transcript_worker(worker_id: int, video_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                                 youtube_analyzer, timing_stats: dict, job_id: str):
    """Worker that fetches individual video transcripts with global rate limiting"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id):
                logger.info(f"🎬 Video Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await video_queue.get()
            video_id = item['video_id']
            start_time = time.time()
            
            logger.info(f"🎬 Video Worker {worker_id}: Fetching transcript for video {video_id}")
            
            # Fetch transcript for this individual video (rate-limited globally)
            transcript = await youtube_analyzer.get_transcript_async(video_id)
            
            transcript_time = time.time() - start_time
            timing_stats['transcript_times'].append(transcript_time)
            
            if transcript:
                logger.info(f"✅ Video Worker {worker_id}: Got transcript for {video_id} in {transcript_time:.2f}s")
                
                # Update video progress tracking
                analysis_results[job_id]['video_progress']['videos_with_transcripts'] += 1
                
                # Update pipeline stage: transcript fetched -> queued for LLM
                update_pipeline_stage(job_id, 'queued_for_llm', 'llm_processing')
                
                # Pass to LLM queue
                await llm_queue.put({
                    'url': item['url'],  # Original channel URL
                    'video_id': video_id,
                    'video_title': item['video_title'],
                    'video_url': item['video_url'],
                    'transcript': transcript,
                    'channel_id': item['channel_id'],
                    'channel_name': item['channel_name'],
                    'channel_handle': item['channel_handle'],
                    'start_time': time.time()
                })
            else:
                logger.warning(f"❌ Video Worker {worker_id}: No transcript available for video {video_id}")
                # Don't count this as a failure since we got some videos from the channel
            
            video_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            video_id = item.get('video_id', 'unknown') if 'item' in locals() else 'unknown'
            logger.error(f"💥 Video transcript worker {worker_id} error processing {video_id}: {e}")
            video_queue.task_done()

async def llm_worker(worker_id: int, llm_queue: asyncio.Queue, result_queue: asyncio.Queue,
                     llm_analyzer, timing_stats: dict, job_id: str):
    """Worker that processes individual videos through LLM"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id):
                logger.info(f"🤖 LLM Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await llm_queue.get()
            url = item['url']  # Original channel URL
            video_id = item['video_id']
            transcript = item['transcript']
            start_time = time.time()
            
            # Update pipeline stage: queued for LLM -> LLM processing
            update_pipeline_stage(job_id, 'queued_for_llm', 'llm_processing')
            
            logger.info(f"🤖 LLM Worker {worker_id}: Processing video {video_id}")
            
            # Create video data structure for LLM analysis
            video_data = {
                'id': video_id,
                'title': item['video_title'],
                'url': item['video_url'],
                'transcript': transcript
            }
            
            # Analyze this single video
            analysis_result = await llm_analyzer.analyze_video_content_async(video_data)
            
            llm_time = time.time() - start_time
            timing_stats['llm_times'].append(llm_time)
            
            # Update video progress tracking
            analysis_results[job_id]['video_progress']['videos_analyzed_by_llm'] += 1
            
            logger.info(f"✅ LLM Worker {worker_id}: Completed video {video_id} in {llm_time:.2f}s")
            
            # Update pipeline stage: LLM processing -> queued for results
            update_pipeline_stage(job_id, 'llm_processing', 'queued_for_results')
            
            # Pass to result queue with all original channel info
            await result_queue.put({
                'url': url,  # Original channel URL
                'video_id': video_id,
                'video_title': item['video_title'],
                'video_url': item['video_url'],
                'video_analysis': analysis_result,
                'transcript': transcript,
                'channel_id': item['channel_id'],
                'channel_name': item['channel_name'],
                'channel_handle': item['channel_handle'],
                'start_time': time.time()
            })
            
            llm_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            video_id = item.get('video_id', 'unknown')
            url = item.get('url', 'unknown')
            channel_name = item.get('channel_name', 'Unknown')
            error_msg = f"Failed to analyze video with AI. Video '{video_id}' from channel '{channel_name}' transcript was retrieved successfully, but AI processing failed: {str(e)}"
            
            logger.error(f"💥 LLM worker {worker_id} error processing video {video_id}: {e}")
            
            # Update pipeline stage: LLM processing -> failed
            update_pipeline_stage(job_id, 'llm_processing', 'failed')
            
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'llm_processing_failed',
                'channel_name': channel_name,
                'video_id': video_id
            })
            llm_queue.task_done()

async def result_worker(worker_id: int, result_queue: asyncio.Queue, timing_stats: dict, job_id: str):
    """Worker that aggregates video results by channel and stores them"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id):
                logger.info(f"📝 Result Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await result_queue.get()
            url = item['url']  # Original channel URL
            start_time = time.time()
            
            # Update pipeline stage: queued for results -> result processing
            update_pipeline_stage(job_id, 'queued_for_results', 'result_processing')
            
            logger.info(f"📝 Result Worker {worker_id}: Processing video result for {url}")
            
            # Get or create channel entry
            if url not in analysis_results[job_id]['results']:
                # Initialize channel entry
                channel_name = item.get('channel_name', '')
                channel_handle = item.get('channel_handle', '')
                
                # Clean channel name/handle
                if isinstance(channel_name, str):
                    channel_name = channel_name.replace('@', '')
                if isinstance(channel_handle, str):
                    channel_handle = channel_handle.replace('@', '')
                    
                if not channel_name and item.get('channel_id'):
                    channel_name = f"Channel {item['channel_id']}"
                
                analysis_results[job_id]['results'][url] = {
                    "url": str(url),
                    "channel_id": item.get('channel_id', "unknown"),
                    "channel_name": channel_name or "Unknown",
                    "channel_handle": channel_handle or "Unknown",
                    "video_analyses": [],
                    "summary": {},
                    "original_videos": []
                }
            
            # Add this video's analysis to the channel
            video_analysis_entry = {
                "video_id": item['video_id'],
                "video_title": item['video_title'],
                "video_url": item['video_url'],
                "analysis": item['video_analysis']
            }
            
            video_data_entry = {
                "id": item['video_id'],
                "title": item['video_title'],
                "url": item['video_url'],
                "transcript": item['transcript']
            }
            
            analysis_results[job_id]['results'][url]["video_analyses"].append(video_analysis_entry)
            analysis_results[job_id]['results'][url]["original_videos"].append(video_data_entry)
            
            # Update video progress tracking
            analysis_results[job_id]['video_progress']['videos_completed'] += 1
            
            # Update progress tracking (count when we complete a video, not when we finish a channel)
            # Note: This will over-count, but we'll fix in final aggregation
            
            # Update pipeline stage: result processing -> completed
            update_pipeline_stage(job_id, 'result_processing', 'completed')
            
            result_time = time.time() - start_time
            logger.info(f"✅ Result Worker {worker_id}: Added video {item['video_id']} to channel {url} in {result_time:.3f}s")
            
            result_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"💥 Result worker {worker_id} error: {e}")
            
            # Update pipeline stage: result processing -> failed (if we can determine the URL)
            if 'item' in locals() and 'url' in item:
                update_pipeline_stage(job_id, 'result_processing', 'failed')
            
            result_queue.task_done()

async def monitor_pipeline_detailed(job_id: str, channel_queue: asyncio.Queue, video_queue: asyncio.Queue, 
                                   llm_queue: asyncio.Queue, result_queue: asyncio.Queue, total_urls: int, timing_stats: dict):
    """Monitor queue depths and progress with enhanced tracking"""
    monitor_start = time.time()
    monitoring_interval = 3  # Monitor every 3 seconds for better granularity
    last_detailed_log = 0
    
    logger.info(f"🔍 Starting enhanced pipeline monitoring for job {job_id}")
    
    while True:
        try:
            await asyncio.sleep(monitoring_interval)
            
            current_time = time.time()
            elapsed = current_time - monitor_start
            
            channel_size = channel_queue.qsize()
            video_size = video_queue.qsize() 
            llm_size = llm_queue.qsize()
            result_size = result_queue.qsize()
            
            completed = len(analysis_results[job_id]['results'])
            failed = len(analysis_results[job_id]['failed_urls'])
            processed = completed + failed
            
            # Always log basic status
            progress_pct = (processed/total_urls*100) if total_urls > 0 else 0
            logger.info(f"📊 T:{channel_size:2d} | V:{video_size:2d} | L:{llm_size:2d} | R:{result_size:2d} | {processed:3d}/{total_urls} ({progress_pct:5.1f}%) | {elapsed:6.1f}s elapsed")
            
            # Update queue depths for analysis
            timing_stats['queue_depths']['channel'].append(channel_size)
            timing_stats['queue_depths']['video'].append(video_size)
            timing_stats['queue_depths']['llm'].append(llm_size)
            timing_stats['queue_depths']['result'].append(result_size)
            timing_stats['timestamps'].append(current_time)
            
            # Detailed logging every 15 seconds
            if elapsed - last_detailed_log >= 15:
                last_detailed_log = elapsed
                
                if processed > 0:
                    rate = processed / elapsed
                    eta_seconds = (total_urls - processed) / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    # Debug current max values
                    current_max_c = max(timing_stats['queue_depths']['channel'], default=0)
                    current_max_v = max(timing_stats['queue_depths']['video'], default=0)
                    current_max_l = max(timing_stats['queue_depths']['llm'], default=0)
                    current_max_r = max(timing_stats['queue_depths']['result'], default=0)
                    
                    logger.info(f"📈 DETAILED STATUS:")
                    logger.info(f"   └─ Processing rate: {rate:.2f} URLs/sec")
                    logger.info(f"   └─ ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)")
                    logger.info(f"   └─ Queue depths - Max seen: C:{current_max_c} | V:{current_max_v} | L:{current_max_l} | R:{current_max_r}")
                    logger.info(f"   └─ Debug: Stats collected: {len(timing_stats['queue_depths']['channel'])} data points")
                    logger.info(f"   └─ Debug: Recent C depths={timing_stats['queue_depths']['channel'][-5:]} (last 5)")
                    logger.info(f"   └─ Debug: Recent V depths={timing_stats['queue_depths']['video'][-5:]} (last 5)")
                    logger.info(f"   └─ Debug: Recent L depths={timing_stats['queue_depths']['llm'][-5:]} (last 5)")
                    logger.info(f"   └─ Debug: Recent R depths={timing_stats['queue_depths']['result'][-5:]} (last 5)")
            
            # If all work is done, break
            if processed >= total_urls and channel_size == 0 and video_size == 0 and llm_size == 0 and result_size == 0:
                logger.info("🏁 All work completed, stopping monitor")
                logger.info(f"📊 Final monitoring stats: {len(timing_stats['queue_depths']['channel'])} data points collected over {elapsed:.1f}s")
                break
                
        except asyncio.CancelledError:
            logger.info("🔍 Pipeline monitor cancelled")
            break
        except Exception as e:
            logger.error(f"💥 Monitor error: {e}")