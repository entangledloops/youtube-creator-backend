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
from src.version import __version__, __build_date__, VERSION_HISTORY
from src.rate_limiter import youtube_rate_limiter
from src.controversy_screener import screen_creator_for_controversy
from src.job_manager import (
    job_queue, current_job_id, job_queue_lock, active_job_tasks,
    process_job_with_cleanup
)
from src.export_handlers import (
    download_bulk_analysis_csv,
    download_bulk_analysis_evidence,
    download_failed_urls_csv
)

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
        
        # Screen for controversies before proceeding
        logger.info(f"🔍 Screening creator {channel_name} for controversies...")
        is_controversial, controversy_reason = await screen_creator_for_controversy(
            channel_name or "Unknown", 
            channel_handle or "Unknown",
            llm_analyzer
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
                'queued_for_discovery': len(cleaned_urls),   # Channels waiting for video discovery
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
            global current_job_id
            
            if current_job_id is None:
                # No job running, start immediately
                current_job_id = job_id
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

@app.get("/api/bulk-analyze/{job_id}/status")
async def get_bulk_analysis_status_alias(job_id: str):
    """Alias for get_bulk_analysis_status to match frontend expectations"""
    logger.info(f"Status alias request for job_id: {job_id}")
    return await get_bulk_analysis_status(job_id)

async def get_bulk_analysis_status(job_id: str):
    """Get detailed status of a bulk analysis job with pipeline breakdown and ETA"""
    if job_id not in analysis_results:
        # Log available job IDs for debugging
        available_jobs = list(analysis_results.keys())
        logger.warning(f"Job {job_id} not found. Available jobs: {available_jobs}")
        logger.warning(f"Total jobs in memory: {len(analysis_results)}")
        
        # Check if this might be an old job
        if len(available_jobs) == 0:
            logger.warning("No jobs in memory - server may have been restarted")
        
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
        
        # Channel progress should never exceed video progress
        # Use video completion rate as a proxy for channel completion
        channel_progress_percentage = video_progress_percentage
    else:
        # Fallback to channel-level progress when no video data available
        video_progress_percentage = (total_processed / job['total_urls'] * 100) if job['total_urls'] > 0 else 0
        channel_progress_percentage = video_progress_percentage
        effective_processed = total_processed
        effective_total = job['total_urls']
    
    # Calculate channel-level progress (separate from video progress)
    channel_progress_percentage = (total_processed / job['total_urls'] * 100) if job['total_urls'] > 0 else 0
    
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
        "api_version": __version__,  # Add version info
        
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
            "video_percentage": video_progress_percentage
        },
        
        # Detailed pipeline stage breakdown
        "pipeline_stages": job.get('pipeline_stages', {}),
        
        # Controversy check failures (channels that were processed despite check failure)
        "controversy_check_failures": len(job.get('controversy_check_failures', {})),
        
        # ETA and performance information
        "eta": eta_info,
        
        # YouTube rate limiting stats
        "youtube_stats": {
            "total_transcript_requests": youtube_rate_limiter['total_transcript_requests'],
            "total_api_calls": youtube_rate_limiter['total_api_calls'],
            "currently_blocked": youtube_rate_limiter['blocked_until'] is not None and youtube_rate_limiter['blocked_until'] > time.time(),
            "blocked_until": youtube_rate_limiter['blocked_until'] if youtube_rate_limiter['blocked_until'] and youtube_rate_limiter['blocked_until'] > time.time() else None,
            "consecutive_blocks": youtube_rate_limiter['consecutive_blocks'],
            "last_block_time": youtube_rate_limiter['last_block_time']
        },
        
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
    logger.info(f"   📤 Sending to frontend - Channel Progress: {channel_progress_percentage:.1f}%, Video Progress: {video_progress_percentage:.1f}%")
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
        # Add controversy check failures
        "controversy_check_failures": job.get('controversy_check_failures', {}),
        # Add explicit counts for frontend debugging
        "summary_counts": {
            "successful": successful_count,
            "failed": failed_count, 
            "total_processed": total_processed,
            "controversy_check_failures": len(job.get('controversy_check_failures', {}))
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
                        next_job['llm_provider'],
                        analysis_results
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