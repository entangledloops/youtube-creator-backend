"""
FastAPI application for YouTube content compliance analysis
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import logging
import uvicorn
import asyncio
import pandas as pd
import uuid
from datetime import datetime
import json
import time

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.creator_processor import CreatorProcessor, ProcessingConfig

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

# Store for bulk analysis results
analysis_results = {}

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
    request: BulkAnalysisRequest = None
):
    """Bulk analyze multiple YouTube creators from a CSV file.
    The file can be a single column of URLs with or without a header.
    Uses direct concurrency instead of background tasks.
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
        
        logger.info(f"Created job {job_id} with {len(cleaned_urls)} URLs")
        logger.info(f"Job {job_id} stored in analysis_results: {job_id in analysis_results}")
        
        # Always use environment LLM_PROVIDER (ignore frontend request)
        llm_provider = os.getenv("LLM_PROVIDER", "local")
        video_limit = request.video_limit if request else 10
        
        logger.info(f"🔧 Bulk analysis using LLM provider: {llm_provider}")
        
        # Start processing with direct pipeline approach
        asyncio.create_task(
            process_creators_pipeline(job_id, cleaned_urls, video_limit, llm_provider)
        )
        
        # Small delay to ensure job is properly stored before returning
        await asyncio.sleep(0.1)
        
        return {
            "job_id": job_id,
            "total_urls": len(cleaned_urls),
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error in bulk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bulk-analyze/{job_id}/status")
async def get_bulk_analysis_status_alias(job_id: str):
    """Alias for get_bulk_analysis_status to match frontend expectations"""
    logger.info(f"Status alias request for job_id: {job_id}")
    return await get_bulk_analysis_status(job_id)

@app.get("/api/bulk-analyze/{job_id}/results")
async def get_bulk_analysis_results(job_id: str):
    """Get detailed results of a bulk analysis job"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
    
    # Calculate counts for debugging
    successful_count = len(job['results'])
    failed_count = len(job['failed_urls'])
    total_processed = successful_count + failed_count
    
    # Debug logging to help frontend troubleshooting
    logger.info(f"🔍 RESULTS ENDPOINT DEBUG for job {job_id}:")
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
    
    logger.info(f"   📤 Sending to frontend: {successful_count} results, {failed_count} failed_urls")
    return response_data

@app.get("/api/bulk-analyze/{job_id}/csv")
async def download_bulk_analysis_csv(job_id: str):
    """Download bulk analysis results as CSV"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = analysis_results[job_id]
    if job['status'] == 'processing':
        raise HTTPException(status_code=400, detail="Analysis still in progress")
        
    # Get all categories
    categories_df = pd.read_csv("../data/YouTube_Controversy_Categories.csv")
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

async def process_creators_pipeline(job_id: str, urls: List[str], video_limit: int, llm_provider: str):
    """Queue-based pipeline with timing and concurrency monitoring"""
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting pipeline for job {job_id} with {len(urls)} URLs")
        logger.info(f"⚖️ Using 10 transcript workers (balanced for speed vs rate limits)")
        
        # Always use environment LLM_PROVIDER (ignore any frontend input)
        actual_llm_provider = os.getenv("LLM_PROVIDER", "local")
        youtube_analyzer = YouTubeAnalyzer()
        logger.info(f"Creating LLMAnalyzer with provider='{actual_llm_provider}' in pipeline")
        llm_analyzer = LLMAnalyzer(provider=actual_llm_provider)
        
        # Create queues for pipeline stages
        transcript_queue = asyncio.Queue(maxsize=1000)
        llm_queue = asyncio.Queue(maxsize=1000)
        result_queue = asyncio.Queue(maxsize=1000)
        
        # Timing tracking
        timing_stats = {
            'transcript_times': [],
            'llm_times': [],
            'total_times': []
        }
        
        # Start all URLs processing immediately (true concurrency)
        logger.info(f"📋 Queuing all {len(urls)} URLs for processing")
        for url in urls:
            await transcript_queue.put({
                'url': url,
                'video_limit': video_limit,
                'start_time': time.time()
            })
        
        # Start workers
        transcript_workers = []
        llm_workers = []
        result_workers = []
        
        # 10 transcript workers to balance speed vs rate limits
        for i in range(10):
            worker = asyncio.create_task(
                transcript_worker(i, transcript_queue, llm_queue, youtube_analyzer, timing_stats, job_id)
            )
            transcript_workers.append(worker)
        
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
        
        # Monitor progress and queue depths
        monitor_task = asyncio.create_task(
            monitor_pipeline(job_id, transcript_queue, llm_queue, result_queue, len(urls))
        )
        
        # Wait for all work to complete
        logger.info("⏳ Waiting for all transcript work to complete...")
        await transcript_queue.join()
        
        logger.info("⏳ Waiting for all LLM work to complete...")
        await llm_queue.join()
        
        logger.info("⏳ Waiting for all result work to complete...")
        await result_queue.join()
        
        # Cancel workers
        for workers in [transcript_workers, llm_workers, result_workers]:
            for worker in workers:
                worker.cancel()
        monitor_task.cancel()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful = len(analysis_results[job_id]['results'])
        failed = len(analysis_results[job_id]['failed_urls'])
        total_processed = successful + failed
        
        # Log comprehensive final statistics
        logger.info("=" * 80)
        logger.info(f"🏁 FINAL STATISTICS for job {job_id}")
        logger.info(f"📊 URLs submitted: {len(urls)}")
        logger.info(f"📊 URLs processed: {total_processed}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏱️  Total time: {total_time:.2f}s")
        logger.info(f"🚀 Throughput: {total_processed/total_time:.2f} URLs/sec")
        
        # Timing percentiles
        if timing_stats['transcript_times']:
            transcript_times = sorted(timing_stats['transcript_times'])
            logger.info(f"📈 Transcript times - P50: {transcript_times[len(transcript_times)//2]:.2f}s, P99: {transcript_times[int(len(transcript_times)*0.99)]:.2f}s")
        
        if timing_stats['llm_times']:
            llm_times = sorted(timing_stats['llm_times'])
            logger.info(f"📈 LLM times - P50: {llm_times[len(llm_times)//2]:.2f}s, P99: {llm_times[int(len(llm_times)*0.99)]:.2f}s")
        
        # Log any discrepancies
        if total_processed != len(urls):
            logger.error(f"🚨 DISCREPANCY: Expected {len(urls)} URLs, processed {total_processed}")
            logger.error(f"🚨 Missing: {len(urls) - total_processed} URLs")
        
        logger.info("=" * 80)
        
        # Mark job as complete
        analysis_results[job_id]['status'] = 'completed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['processed_urls'] = total_processed
        
    except Exception as e:
        logger.error(f"💥 Error in pipeline processing: {str(e)}")
        if job_id in analysis_results:
            analysis_results[job_id]['status'] = 'failed'
            analysis_results[job_id]['error'] = str(e)
            analysis_results[job_id]['completed_at'] = datetime.now().isoformat()

async def transcript_worker(worker_id: int, transcript_queue: asyncio.Queue, llm_queue: asyncio.Queue, 
                           youtube_analyzer, timing_stats: dict, job_id: str):
    """Worker that pulls transcripts and feeds LLM queue"""
    while True:
        try:
            # Get work item
            item = await transcript_queue.get()
            url = item['url']
            video_limit = item['video_limit']
            start_time = time.time()
            
            logger.info(f"🎬 Worker {worker_id}: Getting transcripts for {url}")
            
            # Pull transcripts
            channel_data = await youtube_analyzer.analyze_channel_async(
                url, 
                video_limit=video_limit,
                use_concurrent=True
            )
            
            transcript_time = time.time() - start_time
            timing_stats['transcript_times'].append(transcript_time)
            
            # Determine specific failure reason for clear frontend messaging
            if not channel_data:
                error_msg = f"Failed to access YouTube channel. The URL may be invalid, private, or the channel may not exist: {url}"
                logger.warning(f"❌ Worker {worker_id}: Invalid channel URL - {url}")
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': error_msg,
                    'error_type': 'invalid_channel'
                })
            elif not channel_data.get('videos'):
                # More specific messaging based on what we found
                channel_name = channel_data.get('channel_name', 'Unknown')
                if channel_data.get('video_count', 0) == 0:
                    error_msg = f"Channel '{channel_name}' has no videos available for analysis"
                else:
                    error_msg = f"Channel '{channel_name}' has {channel_data.get('video_count', 0)} videos, but none have transcripts available for analysis. Videos may be too old, in unsupported languages, or have captions disabled."
                
                logger.warning(f"❌ Worker {worker_id}: No transcripts available for {url} - {error_msg}")
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': error_msg,
                    'error_type': 'no_transcripts',
                    'channel_name': channel_name,
                    'video_count': channel_data.get('video_count', 0)
                })
            else:
                logger.info(f"✅ Worker {worker_id}: Got {len(channel_data['videos'])} videos for {url} in {transcript_time:.2f}s")
                # Pass to LLM queue
                await llm_queue.put({
                    'url': url,
                    'channel_data': channel_data,
                    'start_time': time.time()
                })
            
            transcript_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown') if 'item' in locals() else 'unknown'
            error_msg = f"Failed to retrieve channel data or transcripts. There was a technical error accessing YouTube for this URL: {str(e)}"
            
            logger.error(f"💥 Transcript worker {worker_id} error processing {url}: {e}")
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'transcript_processing_error'
            })
            transcript_queue.task_done()

async def llm_worker(worker_id: int, llm_queue: asyncio.Queue, result_queue: asyncio.Queue,
                     llm_analyzer, timing_stats: dict, job_id: str):
    """Worker that processes transcripts through LLM"""
    while True:
        try:
            # Get work item
            item = await llm_queue.get()
            url = item['url']
            channel_data = item['channel_data']
            start_time = time.time()
            
            logger.info(f"🤖 LLM Worker {worker_id}: Processing {url}")
            
            # Send to LLM
            analysis_result = await llm_analyzer.analyze_channel_content_async(channel_data)
            
            llm_time = time.time() - start_time
            timing_stats['llm_times'].append(llm_time)
            
            logger.info(f"✅ LLM Worker {worker_id}: Completed {url} in {llm_time:.2f}s")
            
            # Pass to result queue
            await result_queue.put({
                'url': url,
                'channel_data': channel_data,
                'analysis_result': analysis_result,
                'start_time': time.time()
            })
            
            llm_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown')
            channel_name = item.get('channel_data', {}).get('channel_name', 'Unknown')
            error_msg = f"Failed to analyze content with AI. Channel '{channel_name}' transcripts were retrieved successfully, but AI processing failed: {str(e)}"
            
            logger.error(f"💥 LLM worker {worker_id} error processing {url}: {e}")
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'llm_processing_failed',
                'channel_name': channel_name
            })
            llm_queue.task_done()

async def result_worker(worker_id: int, result_queue: asyncio.Queue, timing_stats: dict, job_id: str):
    """Worker that processes final results and stores them"""
    while True:
        try:
            # Get work item
            item = await result_queue.get()
            url = item['url']
            channel_data = item['channel_data']
            analysis_result = item['analysis_result']
            start_time = time.time()
            
            logger.info(f"📝 Result Worker {worker_id}: Storing {url}")
            
            # Format results
            channel_name = channel_data.get('channel_name', '')
            channel_handle = channel_data.get('channel_handle', '')
            
            # Clean channel name/handle
            if isinstance(channel_name, str):
                channel_name = channel_name.replace('@', '')
            if isinstance(channel_handle, str):
                channel_handle = channel_handle.replace('@', '')
                
            if not channel_name and channel_data.get('channel_id'):
                channel_name = f"Channel {channel_data['channel_id']}"
            
            # Store final result
            final_result = {
                "url": url,
                "channel_id": channel_data.get('channel_id', "unknown"),
                "channel_name": channel_name or "Unknown",
                "channel_handle": channel_handle or "Unknown", 
                "video_analyses": analysis_result.get('video_analyses', []),
                "summary": analysis_result.get('summary', {}),
                "status": "success"
            }
            
            analysis_results[job_id]['results'][url] = final_result
            
            result_time = time.time() - start_time
            logger.info(f"✅ Result Worker {worker_id}: Stored {url} in {result_time:.3f}s")
            
            result_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"💥 Result worker {worker_id} error: {e}")
            result_queue.task_done()

async def monitor_pipeline(job_id: str, transcript_queue: asyncio.Queue, llm_queue: asyncio.Queue, 
                          result_queue: asyncio.Queue, total_urls: int):
    """Monitor queue depths and progress"""
    while True:
        try:
            await asyncio.sleep(5)  # Monitor every 5 seconds
            
            transcript_size = transcript_queue.qsize()
            llm_size = llm_queue.qsize() 
            result_size = result_queue.qsize()
            
            completed = len(analysis_results[job_id]['results'])
            failed = len(analysis_results[job_id]['failed_urls'])
            processed = completed + failed
            
            logger.info(f"📊 PIPELINE STATUS - Queues: T:{transcript_size} | L:{llm_size} | R:{result_size} | Progress: {processed}/{total_urls} ({processed/total_urls*100:.1f}%)")
            
            # If all work is done, break
            if processed >= total_urls and transcript_size == 0 and llm_size == 0 and result_size == 0:
                logger.info("🏁 All work completed, stopping monitor")
                break
                
        except asyncio.CancelledError:
            break
        except Exception as e:
                          logger.error(f"💥 Monitor error: {e}")