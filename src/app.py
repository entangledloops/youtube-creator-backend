"""
FastAPI application for YouTube content compliance analysis

Environment Variables for YouTube Rate Limiting (to prevent IP blocking):
- YOUTUBE_MAX_CONCURRENT: Maximum concurrent transcript requests (default: 2)
- YOUTUBE_REQUEST_DELAY: Delay between transcript requests in seconds (default: 3.0)
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
import re

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.version import __version__, __build_date__, VERSION_HISTORY
from src.rate_limiter import youtube_rate_limiter
from src.controversy_screener import screen_creator_for_controversy
from src import job_manager
from src.job_manager import (
    job_queue, job_queue_lock, active_job_tasks,
    process_job_with_cleanup, update_queue_wait_times
)
from src.export_handlers import (
    download_bulk_analysis_csv,
    download_bulk_analysis_evidence,
    download_failed_urls_csv
)
from src.pipeline_workers import calculate_job_eta
from src.video_cache import video_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set specific loggers to reduce verbosity
logging.getLogger("src.youtube_analyzer").setLevel(logging.WARNING)
logging.getLogger("src.llm_analyzer").setLevel(logging.WARNING)

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
    version=__version__
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

# Background task for monitoring abandoned jobs
abandoned_job_monitor_task = None
ABANDONED_JOB_TIMEOUT = 120  # seconds (2 minutes)

async def monitor_abandoned_jobs():
    """Background task to monitor and cancel abandoned jobs"""
    while True:
        try:
            current_time = datetime.now()
            
            # Check all jobs in analysis_results
            for job_id, job in list(analysis_results.items()):
                # Skip completed, cancelled, or failed jobs
                if job['status'] in ['completed', 'cancelled', 'failed']:
                    continue
                    
                # Get last status check time
                last_check = job.get('last_status_check')
                if not last_check:
                    continue
                    
                # Convert string timestamp to datetime if needed
                if isinstance(last_check, str):
                    last_check = datetime.fromisoformat(last_check)
                    
                # Check if job has been abandoned
                time_since_last_check = (current_time - last_check).total_seconds()
                if time_since_last_check > ABANDONED_JOB_TIMEOUT:
                    logger.warning(f"🕒 Job {job_id} appears abandoned (no status checks for {time_since_last_check:.1f}s)")
                    logger.warning(f"🕒 Job status: {job['status']}, last check: {last_check}")
                    
                    # Cancel the job
                    try:
                        await cancel_bulk_analysis(job_id)
                        logger.info(f"✅ Automatically cancelled abandoned job {job_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to auto-cancel abandoned job {job_id}: {str(e)}")
                else:
                    # Debug: Log active jobs periodically
                    if time_since_last_check > 10:  # Only log if it's been more than 10 seconds
                        logger.debug(f"🔄 Active job {job_id}: {job['status']}, last check {time_since_last_check:.1f}s ago")
            
            # Sleep for a bit before next check
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in abandoned job monitor: {str(e)}")
            await asyncio.sleep(5)  # Sleep before retrying

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    global abandoned_job_monitor_task
    abandoned_job_monitor_task = asyncio.create_task(monitor_abandoned_jobs())

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on app shutdown"""
    if abandoned_job_monitor_task:
        abandoned_job_monitor_task.cancel()
        try:
            await abandoned_job_monitor_task
        except asyncio.CancelledError:
            pass

@app.get("/")
async def root():
    return {
        "message": "YouTube Content Compliance Analyzer API",
        "version": __version__,
        "api_docs": "/docs"
    }

@app.get("/api/version")
async def get_version():
    """Get API version information"""
    return {
        "version": __version__,
        "api_name": "YouTube Content Compliance Analyzer",
        "build_date": __build_date__,
        "version_history": VERSION_HISTORY
    }

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get video cache statistics"""
    try:
        stats = video_cache.get_cache_stats()
        return {
            "cache_stats": stats,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Extract channel info first
        channel_id, channel_name, channel_handle = youtube_analyzer.extract_channel_info_from_url(creator_url)
        
        if not channel_id:
            raise HTTPException(status_code=404, detail="Could not extract channel information from URL")
        
        # Screen for controversies
        logger.info(f"🔍 Screening creator {channel_name} for controversies...")
        is_controversial, controversy_reason, controversy_status = await screen_creator_for_controversy(
            channel_name or "Unknown", 
            channel_handle or "Unknown",
            llm_analyzer,
            creator_url  # Pass the channel URL for better screening
        )
        
        if is_controversial:
            logger.warning(f"🚫 Creator {channel_name} flagged for controversy: {controversy_reason}")
            
            # Return a response indicating the creator was flagged
            return {
                "channel_id": channel_id,
                "channel_name": channel_name or "Unknown",
                "channel_handle": channel_handle or "Unknown",
                "video_analyses": [],
                "summary": {
                    "controversy_flagged": {
                        "flagged": True,
                        "reason": controversy_reason,
                        "status": controversy_status,
                        "message": "This creator has been flagged for ongoing controversies and cannot be analyzed at this time."
                    }
                },
                "status": "controversy_flagged"
            }
        
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
        
        return analysis_result
    except Exception as e:
        logger.error(f"Error analyzing creator: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

def is_youtube_url(url: str) -> bool:
    """Check if a string is a valid YouTube URL"""
    if not isinstance(url, str):
        return False
    
    url = url.strip()
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/',
        r'(?:https?://)?(?:www\.)?youtu\.be/',
        r'(?:https?://)?(?:m\.)?youtube\.com/',
    ]
    
    return any(re.match(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)

def extract_youtube_urls_from_csv(file) -> List[str]:
    """
    Intelligently extract YouTube URLs from a CSV file.
    Handles:
    1. Files with or without headers
    2. Multiple columns - finds the column with the most YouTube URLs
    3. Mixed content - filters out non-URL entries
    """
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        # First, try to read with header detection
        try:
            # Sample first few rows to detect if there's a header
            sample = pd.read_csv(file, nrows=5)
            file.seek(0)
            
            # Check if first row looks like a header (non-URL strings)
            first_row = sample.iloc[0] if len(sample) > 0 else pd.Series()
            has_header = not any(is_youtube_url(str(val)) for val in first_row)
            
            # Read with appropriate header setting
            if has_header:
                df = pd.read_csv(file)
                logger.info(f"📄 CSV detected with header: {list(df.columns)}")
            else:
                df = pd.read_csv(file, header=None)
                logger.info(f"📄 CSV detected without header, {len(df.columns)} columns")
                
        except Exception as e:
            # Fallback: try without header
            file.seek(0)
            df = pd.read_csv(file, header=None)
            logger.info(f"📄 CSV fallback reading without header: {len(df.columns)} columns")
        
        # If only one column, use it directly
        if len(df.columns) == 1:
            urls = df.iloc[:, 0].tolist()
            logger.info(f"📄 Using single column with {len(urls)} rows")
        else:
            # Multiple columns - find the one with the most YouTube URLs
            logger.info(f"📄 Multiple columns detected ({len(df.columns)}), analyzing content...")
            
            best_column = None
            max_youtube_count = 0
            
            for col_idx, column in enumerate(df.columns):
                # Count YouTube URLs in this column
                youtube_count = sum(1 for val in df[column] if is_youtube_url(str(val)))
                logger.info(f"   📊 Column {col_idx} ('{column}'): {youtube_count} YouTube URLs")
                
                if youtube_count > max_youtube_count:
                    max_youtube_count = youtube_count
                    best_column = column
            
            if best_column is not None and max_youtube_count > 0:
                urls = df[best_column].tolist()
                logger.info(f"✅ Selected column '{best_column}' with {max_youtube_count} YouTube URLs")
            else:
                # No column has YouTube URLs, use first column as fallback
                urls = df.iloc[:, 0].tolist()
                logger.warning(f"⚠️ No columns contain YouTube URLs, using first column as fallback")
        
        # Clean and validate URLs
        cleaned_urls = []
        skipped_count = 0
        
        for url in urls:
            url_str = str(url).strip()
            
            # Skip empty, NaN, or clearly non-URL values
            if not url_str or url_str.lower() in ['nan', 'none', 'null', '']:
                continue
                
            if is_youtube_url(url_str):
                # Ensure URL has protocol
                if not url_str.startswith(('http://', 'https://')):
                    url_str = 'https://' + url_str
                cleaned_urls.append(url_str)
            else:
                skipped_count += 1
                logger.debug(f"Skipping non-YouTube URL: {url_str}")
        
        if skipped_count > 0:
            logger.info(f"📊 Skipped {skipped_count} non-YouTube entries")
        
        logger.info(f"✅ Extracted {len(cleaned_urls)} valid YouTube URLs from CSV")
        return cleaned_urls
        
    except Exception as e:
        logger.error(f"❌ Error processing CSV file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/api/bulk-analyze")
async def bulk_analyze(
    file: UploadFile = File(...),
    request: BulkAnalysisRequest = None
):
    """Bulk analyze multiple YouTube creators from a CSV file with job queuing.
    The file can contain single or multiple columns. The system will intelligently
    detect and use the column with the most YouTube URLs. Headers are optional.
    Jobs are processed sequentially to avoid resource conflicts.
    """
    try:
        # Use improved CSV processing
        cleaned_urls = extract_youtube_urls_from_csv(file.file)
                
        if not cleaned_urls:
            raise HTTPException(status_code=400, detail="No valid YouTube URLs found in file. Please ensure your CSV contains YouTube channel or video URLs.")
            
        logger.info(f"📋 Prepared {len(cleaned_urls)} URLs for bulk analysis")
            
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Always use environment LLM_PROVIDER (ignore frontend request)
        llm_provider = os.getenv("LLM_PROVIDER", "local")
        video_limit = request.video_limit if request else 10
        
        # Initialize results storage with detailed pipeline tracking
        analysis_results[job_id] = {
            'status': 'queued',
            'started_at': datetime.now().isoformat(),
            'last_status_check': datetime.now().isoformat(),
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
                'videos_completed': 0,
                'cache_hits': 0  # Track cache hits
            },
            
            # Detailed pipeline stage tracking
            'pipeline_stages': {
                'queued_for_discovery': 0,                   # Channels waiting for video discovery (will be incremented as URLs are queued)
                'discovering_videos': 0,                     # Channels currently being processed for video discovery
                'queued_for_controversy': 0,                 # Channels waiting for controversy screening
                'screening_controversy': 0,                  # Channels currently being screened for controversy
                'controversy_check_failed': 0,               # Channels where controversy check failed (but continued)
                'queued_for_transcripts': 0,                 # Individual videos waiting for transcript fetch
                'fetching_transcripts': 0,                   # Individual videos currently being transcript fetched
                'queued_for_llm': 0,                        # Videos with transcripts waiting for LLM analysis
                'llm_processing': 0,                        # Videos currently being analyzed by LLM
                'queued_for_results': 0,                    # LLM results waiting for aggregation
                'result_processing': 0,                     # Results being processed into final format
                'completed': 0,                             # Successfully completed videos
                'failed': 0                                 # Failed videos/channels
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
            global job_queue  # Must declare global before using it
            
            # DEBUG: Log current queue state
            logger.info(f"🔍 QUEUE DEBUG: current_job_id = {job_manager.current_job_id}")
            logger.info(f"🔍 QUEUE DEBUG: job_queue length = {len(job_queue)}")
            logger.info(f"🔍 QUEUE DEBUG: active_job_tasks keys = {list(active_job_tasks.keys())}")
            
            if job_manager.current_job_id is None:
                # No job running, start immediately
                job_manager.current_job_id = job_id
                analysis_results[job_id]['status'] = 'processing'
                analysis_results[job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting job {job_id} immediately (no queue)")
                
                # Start processing
                task = asyncio.create_task(
                    process_job_with_cleanup(job_id, cleaned_urls, video_limit, llm_provider, analysis_results)
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
                    if job_manager.current_job_id and job_manager.current_job_id in analysis_results:
                        current_job = analysis_results[job_manager.current_job_id]
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
            # Update queue wait times dynamically
            await update_queue_wait_times(analysis_results)
            
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

@app.get("/api/bulk-analyze/{job_id}/status")
async def get_bulk_analysis_status(job_id: str):
    """Get detailed status of a bulk analysis job with pipeline breakdown and ETA"""
    try:
        # Debug: Log the job lookup attempt
        logger.debug(f"🔍 Looking up job {job_id}")
        logger.debug(f"🔍 Available jobs: {list(analysis_results.keys())}")
        
        # Get job data
        job = analysis_results.get(job_id)
        if not job:
            logger.error(f"❌ Job {job_id} not found in analysis_results")
            logger.error(f"❌ Available job IDs: {list(analysis_results.keys())}")
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Update last status check timestamp
        job['last_status_check'] = datetime.now().isoformat()
        
        # Calculate progress percentages
        total_urls = job['total_urls']
        successful_count = len(job['results'])
        failed_count = len(job['failed_urls'])
        total_processed = successful_count + failed_count
        channel_progress_percentage = (total_processed / total_urls * 100) if total_urls > 0 else 0
        
        # Calculate video progress
        video_progress = job.get('video_progress', {})
        total_videos = video_progress.get('total_videos_discovered', 0)
        videos_with_transcripts = video_progress.get('videos_with_transcripts', 0)
        videos_analyzed = video_progress.get('videos_analyzed_by_llm', 0)
        videos_completed = video_progress.get('videos_completed', 0)
        cache_hits = video_progress.get('cache_hits', 0)
        video_progress_percentage = (videos_completed / total_videos * 100) if total_videos > 0 else 0
        
        # Calculate ETA using shared function
        eta_info = calculate_job_eta(job)
        
        # Get controversy check failures
        controversy_failures = job.get('controversy_check_failures', {})
        controversy_failed_count = sum(1 for f in controversy_failures.values() 
                                     if f.get('status') in ['controversial', 'error'])
        
        # Get pipeline stages
        pipeline_stages = job.get('pipeline_stages', {})
        
        # Update controversy check failed count to match actual failures
        if 'controversy_check_failed' in pipeline_stages:
            pipeline_stages['controversy_check_failed'] = controversy_failed_count
        
        # Get real-time queue sizes from active job tasks
        queue_sizes = {
            'channel_queue_size': 0,
            'controversy_queue_size': 0,
            'video_queue_size': 0,
            'transcript_queue_size': 0,
            'analysis_queue_size': 0
        }
        
        if job_id in active_job_tasks:
            for task in active_job_tasks[job_id]:
                if hasattr(task, 'get_queues'):
                    queues = task.get_queues()
                    if queues:
                        queue_sizes.update({
                            'channel_queue_size': queues.get('channel_queue', asyncio.Queue()).qsize(),
                            'controversy_queue_size': queues.get('controversy_queue', asyncio.Queue()).qsize(),
                            'video_queue_size': queues.get('video_queue', asyncio.Queue()).qsize(),
                            'transcript_queue_size': queues.get('transcript_queue', asyncio.Queue()).qsize(),
                            'analysis_queue_size': queues.get('llm_queue', asyncio.Queue()).qsize()
                        })
        
        # Build detailed response
        response = {
            "job_id": job_id,
            "status": job['status'],
            "started_at": job['started_at'],
            "completed_at": job.get('completed_at'),
            "total_urls": job['total_urls'],
            "processed_urls": job['processed_urls'],
            "api_version": __version__,
            
            # Basic progress info (for backward compatibility)
            "progress": {
                "successful": successful_count,
                "failed": failed_count,
                "total_processed": total_processed,
                "percentage": channel_progress_percentage
            },
            
            # Video-level progress details
            "video_progress": {
                "total_videos_discovered": total_videos,
                "videos_with_transcripts": videos_with_transcripts,
                "videos_analyzed_by_llm": videos_analyzed,
                "videos_completed": videos_completed,
                "video_percentage": video_progress_percentage,
                "cache_hits": cache_hits
            },
            
            # Detailed pipeline stage breakdown
            "pipeline_stages": pipeline_stages,
            
            # Controversy check failures
            "controversy_check_failures": {
                "total": len(controversy_failures),
                "by_status": {
                    "controversial": sum(1 for f in controversy_failures.values() if f.get('status') == 'controversial'),
                    "error": sum(1 for f in controversy_failures.values() if f.get('status') == 'error'),
                    "not_controversial": sum(1 for f in controversy_failures.values() if f.get('status') == 'not_controversial')
                }
            },
            
            # YouTube rate limiting stats
            "youtube_stats": {
                "total_transcript_requests": youtube_rate_limiter['total_transcript_requests'],
                "total_api_calls": youtube_rate_limiter['total_api_calls'],
                "currently_blocked": youtube_rate_limiter['blocked_until'] is not None and youtube_rate_limiter['blocked_until'] > time.time(),
                "blocked_until": youtube_rate_limiter['blocked_until'] if youtube_rate_limiter['blocked_until'] and youtube_rate_limiter['blocked_until'] > time.time() else None,
                "consecutive_blocks": youtube_rate_limiter['consecutive_blocks'],
                "last_block_time": youtube_rate_limiter['last_block_time']
            },
            
            # ETA and performance information
            "eta": eta_info,
            
            # Queue information with real-time sizes
            "queue_info": queue_sizes
        }
        
        # Add queue position info for queued jobs
        if job['status'] == 'queued':
            # Update queue wait times dynamically
            await update_queue_wait_times(analysis_results)
            
            response["queue_info"].update({
                "queue_position": job.get('queue_position', 0),
                "estimated_wait_minutes": job.get('estimated_wait_minutes', 0),
                "estimated_start_time": job.get('estimated_start_time'),
                "message": f"Your job is queued at position {job.get('queue_position', 0)}. Estimated wait time: {job.get('estimated_wait_minutes', 0)} minutes."
            })
        
        # DEBUG: Log what we're sending to frontend
        # logger.info(f"   📤 Sending to frontend - Channel Progress: {channel_progress_percentage:.1f}%, Video Progress: {video_progress_percentage:.1f}%")
        # logger.info(f"   📤 Pipeline stages being sent: {response['pipeline_stages']}")
        # logger.info(f"   📤 Queue sizes: {queue_sizes}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Calculate video-level statistics
    total_videos = 0
    videos_with_transcripts = 0
    videos_analyzed = 0
    videos_completed = 0
    
    for channel_result in job['results'].values():
        total_videos += len(channel_result.get('video_analyses', []))
        for video in channel_result.get('video_analyses', []):
            if video.get('transcript'):
                videos_with_transcripts += 1
            if video.get('analysis'):
                videos_analyzed += 1
            if video.get('analysis') and video.get('transcript'):
                videos_completed += 1
    
    response_data = {
        "job_id": job_id,
        "status": job['status'],
        "started_at": job['started_at'],
        "completed_at": job.get('completed_at'),
        "total_urls": job['total_urls'],
        "processed_urls": job['processed_urls'],
        "results": job['results'],
        "failed_urls": job['failed_urls'],
        # Add controversy check failures
        "controversy_check_failures": job.get('controversy_check_failures', {}),
        # Add explicit counts for frontend debugging
        "summary_counts": {
            "successful": successful_count,
            "failed": failed_count, 
            "total_processed": total_processed,
            "controversy_check_failures": len(job.get('controversy_check_failures', {}))
        },
        # Add video-level statistics
        "video_statistics": {
            "total_videos": total_videos,
            "videos_with_transcripts": videos_with_transcripts,
            "videos_analyzed": videos_analyzed,
            "videos_completed": videos_completed
        }
    }
    
    # Add cancellation info if applicable
    if job['status'] == 'cancelled':
        response_data["cancellation_info"] = {
            "partial_results": True,
            "message": f"Job was cancelled. Partial results available for {successful_count} channels.",
            "statistics": {
                "channels_processed": successful_count,
                "videos_processed": total_videos,
                "completion_percentage": (total_processed / job['total_urls'] * 100) if job['total_urls'] > 0 else 0,
                "video_completion_percentage": (videos_completed / total_videos * 100) if total_videos > 0 else 0
            }
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
        global job_queue  # Must declare global before using it
        
        # DEBUG: Log current queue state
        logger.info(f"🔍 QUEUE DEBUG: current_job_id = {job_manager.current_job_id}")
        logger.info(f"🔍 QUEUE DEBUG: job_queue length = {len(job_queue)}")
        logger.info(f"🔍 QUEUE DEBUG: active_job_tasks keys = {list(active_job_tasks.keys())}")
        
        if job['status'] == 'queued':
            # Remove from queue
            job_queue = [j for j in job_queue if j['job_id'] != job_id]
            
            # Update queue positions for remaining jobs
            for i, queued_job in enumerate(job_queue):
                queued_job_id = queued_job['job_id']
                analysis_results[queued_job_id]['queue_position'] = i + 1
            
            # Mark as cancelled
            analysis_results[job_id]['status'] = 'cancelled'
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"✅ Cancelled queued job {job_id}")
            
        elif job['status'] == 'processing' and job_manager.current_job_id == job_id:
            # Cancel active job
            analysis_results[job_id]['status'] = 'cancelling'
            
            # Calculate current progress before cancellation
            successful_count = len(job['results'])
            failed_count = len(job['failed_urls'])
            total_processed = successful_count + failed_count
            
            # Calculate video-level statistics
            total_videos = 0
            videos_with_transcripts = 0
            videos_analyzed = 0
            videos_completed = 0
            
            for channel_result in job['results'].values():
                total_videos += len(channel_result.get('video_analyses', []))
                for video in channel_result.get('video_analyses', []):
                    if video.get('transcript'):
                        videos_with_transcripts += 1
                    if video.get('analysis'):
                        videos_analyzed += 1
                    if video.get('analysis') and video.get('transcript'):
                        videos_completed += 1
            
            # Cancel all active tasks for this job
            if job_id in active_job_tasks:
                logger.info(f"🛑 Cancelling {len(active_job_tasks[job_id])} active tasks for job {job_id}")
                
                # First, clear all queues to prevent new work from being picked up
                for task in active_job_tasks[job_id]:
                    if hasattr(task, 'clear_queues'):
                        await task.clear_queues()
                
                # Then cancel the tasks
                for task in active_job_tasks[job_id]:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to finish cancelling with a timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*active_job_tasks[job_id], return_exceptions=True),
                        timeout=5.0  # 5 second timeout for graceful shutdown
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for tasks to cancel, forcing shutdown")
                except Exception as e:
                    logger.warning(f"Error during task cancellation: {str(e)}")
                
                # Clean up task tracking
                del active_job_tasks[job_id]
            
            # Mark as cancelled and preserve partial results
            analysis_results[job_id]['status'] = 'cancelled'
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
            
            # Reset current job and start next in queue
            job_manager.current_job_id = None
            
            # Start next job if any are queued
            if job_queue:
                next_job = job_queue.pop(0)
                job_manager.current_job_id = next_job['job_id']
                
                # Update status and start processing
                analysis_results[job_manager.current_job_id]['status'] = 'processing'
                analysis_results[job_manager.current_job_id]['queue_position'] = 0
                
                logger.info(f"🚀 Starting next queued job {job_manager.current_job_id}")
                
                task = asyncio.create_task(
                    process_job_with_cleanup(
                        job_manager.current_job_id,
                        next_job['urls'],
                        next_job['video_limit'],
                        next_job['llm_provider'],
                        analysis_results
                    )
                )
                
                # Track task for cancellation
                active_job_tasks[job_manager.current_job_id] = [task]
            
            logger.info(f"✅ Cancelled active job {job_id}")
    
    # Calculate final statistics for the cancelled job
    successful_count = len(job['results'])
    failed_count = len(job['failed_urls'])
    total_processed = successful_count + failed_count
    
    # Calculate video-level statistics
    total_videos = 0
    videos_with_transcripts = 0
    videos_analyzed = 0
    videos_completed = 0
    
    for channel_result in job['results'].values():
        total_videos += len(channel_result.get('video_analyses', []))
        for video in channel_result.get('video_analyses', []):
            if video.get('transcript'):
                videos_with_transcripts += 1
            if video.get('analysis'):
                videos_analyzed += 1
            if video.get('analysis') and video.get('transcript'):
                videos_completed += 1
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled successfully",
        "partial_results": {
            "successful_channels": successful_count,
            "failed_urls": failed_count,
            "total_processed": total_processed,
            "results_preserved": successful_count > 0,
            "completion_percentage": (total_processed / job['total_urls'] * 100) if job['total_urls'] > 0 else 0
        },
        "video_statistics": {
            "total_videos": total_videos,
            "videos_with_transcripts": videos_with_transcripts,
            "videos_analyzed": videos_analyzed,
            "videos_completed": videos_completed,
            "completion_percentage": (videos_completed / total_videos * 100) if total_videos > 0 else 0
        },
        "cancelled_at": analysis_results[job_id]['completed_at'],
        "next_steps": "You can retrieve the partial results using the GET /api/bulk-analyze/{job_id}/results endpoint"
    }

@app.get("/api/bulk-analyze/{job_id}/csv")
async def download_csv(job_id: str):
    """Download bulk analysis results as CSV"""
    return await download_bulk_analysis_csv(job_id, analysis_results)

@app.get("/api/bulk-analyze/{job_id}/evidence")
async def download_evidence(job_id: str):
    """Download detailed evidence and transcripts as JSON"""
    return await download_bulk_analysis_evidence(job_id, analysis_results)

@app.get("/api/bulk-analyze/{job_id}/failed-csv")
async def download_failed_csv(job_id: str):
    """Download failed URLs from bulk analysis as CSV"""
    return await download_failed_urls_csv(job_id, analysis_results)

@app.delete("/api/bulk-analyze/cancel-all")
async def cancel_all_jobs():
    """Cancel all ongoing jobs and clear the job queue"""
    logger.info("🔴 CANCEL-ALL ENDPOINT: Request received")
    try:
        cancelled_jobs = []
        
        async with job_queue_lock:
            global job_queue
            
            # Cancel the currently running job if any
            if job_manager.current_job_id and job_manager.current_job_id in analysis_results:
                current_job = analysis_results[job_manager.current_job_id]
                if current_job['status'] in ['processing', 'queued']:
                    logger.info(f"🛑 CANCEL ALL: Cancelling active job {job_manager.current_job_id}")
                    
                    # Mark as cancelled
                    analysis_results[job_manager.current_job_id]['status'] = 'cancelled'
                    analysis_results[job_manager.current_job_id]['completed_at'] = datetime.now().isoformat()
                    
                    # Cancel all active tasks for this job
                    if job_manager.current_job_id in active_job_tasks:
                        logger.info(f"🛑 Cancelling {len(active_job_tasks[job_manager.current_job_id])} active tasks for job {job_manager.current_job_id}")
                        
                        # First, clear all queues to prevent new work from being picked up
                        for task in active_job_tasks[job_manager.current_job_id]:
                            if hasattr(task, 'clear_queues'):
                                await task.clear_queues()
                        
                        # Then cancel the tasks
                        for task in active_job_tasks[job_manager.current_job_id]:
                            if not task.done():
                                task.cancel()
                        
                        # Wait for tasks to finish cancelling with a timeout
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*active_job_tasks[job_manager.current_job_id], return_exceptions=True),
                                timeout=3.0  # 3 second timeout for cancel-all
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout waiting for tasks to cancel for job {job_manager.current_job_id}")
                        except Exception as e:
                            logger.warning(f"Error during task cancellation for job {job_manager.current_job_id}: {str(e)}")
                        
                        # Clean up task tracking
                        del active_job_tasks[job_manager.current_job_id]
                    
                    cancelled_jobs.append({
                        'job_id': job_manager.current_job_id,
                        'status': 'was_processing',
                        'partial_results': len(current_job['results']),
                        'failed_urls': len(current_job['failed_urls'])
                    })
            
            # Cancel all queued jobs
            queued_jobs_cancelled = []
            for queued_job in job_queue:
                queued_job_id = queued_job['job_id']
                if queued_job_id in analysis_results:
                    logger.info(f"🛑 CANCEL ALL: Cancelling queued job {queued_job_id}")
                    
                    # Mark as cancelled
                    analysis_results[queued_job_id]['status'] = 'cancelled'
                    analysis_results[queued_job_id]['completed_at'] = datetime.now().isoformat()
                    
                    queued_jobs_cancelled.append({
                        'job_id': queued_job_id,
                        'status': 'was_queued',
                        'queue_position': analysis_results[queued_job_id].get('queue_position', 0)
                    })
            
            cancelled_jobs.extend(queued_jobs_cancelled)
            
            # Clear the job queue and reset current job
            job_queue.clear()
            job_manager.current_job_id = None
            
            logger.info(f"✅ CANCEL ALL: Cancelled {len(cancelled_jobs)} jobs total")
        
        return {
            "message": f"Successfully cancelled {len(cancelled_jobs)} jobs",
            "cancelled_jobs": cancelled_jobs,
            "summary": {
                "total_cancelled": len(cancelled_jobs),
                "active_jobs_cancelled": len([j for j in cancelled_jobs if j['status'] == 'was_processing']),
                "queued_jobs_cancelled": len([j for j in cancelled_jobs if j['status'] == 'was_queued']),
                "queue_cleared": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cancelling all jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 