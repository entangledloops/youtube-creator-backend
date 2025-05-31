"""
Pipeline worker functions for processing YouTube channels through various stages
"""
import asyncio
import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import traceback
import re
import os

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.rate_limiter import youtube_rate_limiter
from src.controversy_screener import screen_creator_for_controversy
from src.video_cache import video_cache
from src.export_handlers import all_categories_glob

logger = logging.getLogger(__name__)

# Define worker counts
MAX_DISCOVERY_WORKERS = 2
MAX_CONTROVERSY_WORKERS = 1 # Keep this low as it can be intensive
MAX_TRANSCRIPT_WORKERS = 5
MAX_LLM_WORKERS = 3 # Increased from 2

# Shared lock for accessing analysis_results dictionary
job_data_lock = asyncio.Lock()

def calculate_job_eta(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate ETA information for a job based on current progress.
    This is the single source of truth for ETA calculations.
    
    Args:
        job: Dictionary containing job information including status, progress, and timing data
        
    Returns:
        Dictionary containing ETA information including:
        - estimated_completion_time: ISO format timestamp
        - estimated_minutes_remaining: Integer minutes
        - processing_rate_per_minute: Float rate
        - elapsed_minutes: Float minutes
        - progress_type: Either "videos" or "channels"
    """
    eta_info = {}
    
    if job['status'] == 'processing':
        elapsed_time = time.time() - job['performance_stats']['overall_start_time']
        
        # Calculate processing rate using video completion
        video_progress = job.get('video_progress', {})
        videos_completed = video_progress.get('videos_completed', 0)
        total_videos = video_progress.get('total_videos_discovered', 0)
        
        if videos_completed > 0:
            # Calculate processing rate using video completion
            processing_rate_per_second = videos_completed / elapsed_time
            processing_rate_per_minute = processing_rate_per_second * 60
            
            # Estimate remaining time
            remaining_videos = total_videos - videos_completed
            estimated_seconds_remaining = remaining_videos / processing_rate_per_second if processing_rate_per_second > 0 else 0
            estimated_minutes_remaining = estimated_seconds_remaining / 60
            
            eta_info = {
                "estimated_completion_time": (
                    datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                ).isoformat(),
                "estimated_minutes_remaining": max(1, int(estimated_minutes_remaining)),
                "processing_rate_per_minute": round(processing_rate_per_minute, 2),
                "elapsed_minutes": round(elapsed_time / 60, 1),
                "progress_type": "videos"
            }
        else:
            # Fallback to channel-level progress if no videos completed yet
            total_processed = len(job['results']) + len(job['failed_urls'])
            if total_processed > 0:
                processing_rate_per_second = total_processed / elapsed_time
                processing_rate_per_minute = processing_rate_per_second * 60
                
                # Estimate remaining time
                remaining_urls = job['total_urls'] - total_processed
                estimated_seconds_remaining = remaining_urls / processing_rate_per_second if processing_rate_per_second > 0 else 0
                estimated_minutes_remaining = estimated_seconds_remaining / 60
                
                eta_info = {
                    "estimated_completion_time": (
                        datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                    ).isoformat(),
                    "estimated_minutes_remaining": max(1, int(estimated_minutes_remaining)),
                    "processing_rate_per_minute": round(processing_rate_per_minute, 2),
                    "elapsed_minutes": round(elapsed_time / 60, 1),
                    "progress_type": "channels"
                }
            else:
                eta_info = {
                    "estimated_completion_time": None,
                    "estimated_minutes_remaining": None,
                    "processing_rate_per_minute": 0,
                    "elapsed_minutes": round(elapsed_time / 60, 1),
                    "progress_type": "channels"
                }
    
    return eta_info

def update_pipeline_stage(job_id: str, from_stage: str, to_stage: str, count: int = 1, analysis_results: dict = None):
    """Helper function to update pipeline stage counters"""
    if not analysis_results or job_id not in analysis_results:
        return
    
    stages = analysis_results[job_id]['pipeline_stages']
    
    # Decrease from_stage count
    if from_stage and from_stage in stages:
        stages[from_stage] = max(0, stages[from_stage] - count)
    
    # Increase to_stage count
    if to_stage and to_stage in stages:
        stages[to_stage] = stages[to_stage] + count

def is_job_cancelled(job_id: str, analysis_results: dict = None) -> bool:
    """Check if a job has been cancelled"""
    if not analysis_results or job_id not in analysis_results:
        return True
    return analysis_results[job_id]['status'] in ['cancelled', 'cancelling']

async def channel_discovery_worker(worker_id: int, channel_queue: asyncio.Queue, controversy_queue: asyncio.Queue,
                                  youtube_analyzer, timing_stats: dict, job_id: str, analysis_results: dict):
    """Worker that discovers videos in channels and feeds individual videos to video queue"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"📋 Channel Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await channel_queue.get()
            url = item['url']
            video_limit = item['video_limit']
            start_time = time.time()
            
            # Update pipeline stage: queued -> discovering
            update_pipeline_stage(job_id, 'queued_for_discovery', 'discovering_videos', analysis_results=analysis_results)
            
            logger.debug(f"📋 Channel Worker {worker_id}: Discovering videos for {url}")
            
            # Get channel info and video list (without transcripts yet) - RUN IN THREAD TO AVOID BLOCKING
            channel_id, channel_name, channel_handle = await asyncio.to_thread(
                youtube_analyzer.extract_channel_info_from_url, url
            )
            
            if not channel_id:
                error_msg = f"Failed to access YouTube channel. The URL may be invalid, private, or the channel may not exist: {url}"
                logger.warning(f"❌ Channel Worker {worker_id}: Invalid channel URL - {url}")
                
                # Update pipeline stage: discovering -> failed
                update_pipeline_stage(job_id, 'discovering_videos', 'failed', analysis_results=analysis_results)
                
                analysis_results[job_id]['failed_urls'].append({
                    'url': url,
                    'error': error_msg,
                    'error_type': 'invalid_channel'
                })
                logger.info(f"🔍 FAILED_URLS DEBUG: Added {url} to failed_urls (invalid_channel) - total failed: {len(analysis_results[job_id]['failed_urls'])}")
                
                # Mark task as done
                channel_queue.task_done()
                logger.debug(f"📋 Channel Worker {worker_id}: Marked invalid channel as done (queue size: {channel_queue.qsize()})")
            else:
                # Get video list from channel - RUN IN THREAD TO AVOID BLOCKING
                videos = await asyncio.to_thread(
                    youtube_analyzer.get_videos_from_channel, channel_id, limit=video_limit
                )
                
                if not videos:
                    error_msg = f"Channel '{channel_name or 'Unknown'}' has no videos available for analysis"
                    logger.warning(f"❌ Channel Worker {worker_id}: No videos found for {url}")
                    
                    # Update pipeline stage: discovering -> failed
                    update_pipeline_stage(job_id, 'discovering_videos', 'failed', analysis_results=analysis_results)
                    
                    analysis_results[job_id]['failed_urls'].append({
                        'url': url,
                        'error': error_msg,
                        'error_type': 'no_videos',
                        'channel_name': channel_name or 'Unknown'
                    })
                    logger.info(f"🔍 FAILED_URLS DEBUG: Added {url} to failed_urls (no_videos) - total failed: {len(analysis_results[job_id]['failed_urls'])}")
                    
                    # Mark task as done
                    channel_queue.task_done()
                    logger.debug(f"📋 Channel Worker {worker_id}: Marked no-videos channel as done (queue size: {channel_queue.qsize()})")
                else:
                    discovery_time = time.time() - start_time
                    timing_stats['channel_discovery'].append(discovery_time)
                    
                    logger.debug(f"✅ Channel Worker {worker_id}: Found {len(videos)} videos for {url} in {discovery_time:.2f}s")
                    
                    # Update pipeline stage: discovering -> queued for controversy
                    update_pipeline_stage(job_id, 'discovering_videos', 'queued_for_controversy', analysis_results=analysis_results)
                    
                    # Pass channel data to controversy screening queue
                    await controversy_queue.put({
                        'url': url,
                        'channel_id': channel_id,
                        'channel_name': channel_name,
                        'channel_handle': channel_handle,
                        'videos': videos,
                        'start_time': time.time()
                    })
            
                    # Mark task as done
                    channel_queue.task_done()
                    logger.debug(f"📋 Channel Worker {worker_id}: Queued channel for controversy screening (queue size: {channel_queue.qsize()})")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown') if 'item' in locals() else 'unknown'
            error_msg = f"Failed to retrieve channel data. There was a technical error accessing YouTube for this URL: {str(e)}"
            
            logger.error(f"💥 Channel discovery worker {worker_id} error processing {url}: {e}")
            
            # Update pipeline stage: discovering -> failed
            update_pipeline_stage(job_id, 'discovering_videos', 'failed', analysis_results=analysis_results)
            
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'channel_discovery_error'
            })
            logger.info(f"🔍 FAILED_URLS DEBUG: Added {url} to failed_urls (channel_discovery_error) - total failed: {len(analysis_results[job_id]['failed_urls'])}")
            
            channel_queue.task_done()

async def controversy_screening_worker(worker_id: int, controversy_queue: asyncio.Queue, video_queue: asyncio.Queue,
                                      llm_analyzer, timing_stats: dict, job_id: str, analysis_results: dict):
    """Worker that screens channels for controversy before processing videos"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"⚠️ Controversy Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
            
            # Get work item
            item = await controversy_queue.get()
            url = item['url']
            channel_name = item['channel_name']
            channel_handle = item['channel_handle']
            channel_id = item['channel_id']
            videos = item['videos']
            start_time = time.time()
            
            # Update pipeline stage: queued for controversy -> screening controversy
            update_pipeline_stage(job_id, 'queued_for_controversy', 'screening_controversy', analysis_results=analysis_results)
            
            logger.debug(f"⚠️ Controversy Worker {worker_id}: Screening {channel_name} for controversies")
            
            # Screen for controversies
            is_controversial, controversy_reason, controversy_status = await screen_creator_for_controversy(
                channel_name or "Unknown", 
                channel_handle or "Unknown",
                llm_analyzer,
                url  # Pass the channel URL for better screening
            )
            
            screening_time = time.time() - start_time
            timing_stats['controversy_screening'].append(screening_time)
            
            # Initialize controversy check failures if not exists
            if 'controversy_check_failures' not in analysis_results[job_id]:
                analysis_results[job_id]['controversy_check_failures'] = {}
            
            # Add to controversy check failures with status
            if url not in analysis_results[job_id]['controversy_check_failures']:
                analysis_results[job_id]['controversy_check_failures'][url] = {
                    'channel_name': channel_name,
                    'reason': controversy_reason,
                    'status': controversy_status,
                    'timestamp': datetime.now().isoformat()
                }
            
            if controversy_status == 'controversial':
                logger.warning(f"🚫 Controversy Worker {worker_id}: Creator {channel_name} flagged for controversy: {controversy_reason}")
                
                # Update pipeline stage: screening -> controversy check failed
                update_pipeline_stage(job_id, 'screening_controversy', 'controversy_check_failed', analysis_results=analysis_results)
                
                # FIXED: Add controversial channels to RESULTS with proper scoring, not failed_urls
                # Create a proper analysis result structure for controversial channels
                controversy_analysis = {
                    "video_id": "controversy_screening",
                    "video_title": "Controversy Screening Result",
                    "video_url": url,
                    "analysis": {
                        "results": {
                            "Controversy or Cancelled Creators": {
                                "score": 1.0,  # High Risk score
                                "evidence": [controversy_reason],
                                "explanation": f"Channel flagged during controversy screening: {controversy_reason}"
                            }
                        }
                    }
                }
                
                # Create channel summary with controversy category marked
                controversy_summary = {
                    "Controversy or Cancelled Creators": {
                        "max_score": 1.0,
                        "average_score": 1.0,
                        "videos_with_violations": 1,
                        "total_videos": 1,
                        "examples": [{
                            "video_id": "controversy_screening",
                            "video_title": "Controversy Screening Result",
                            "video_url": url,
                            "score": 1.0,
                            "evidence": controversy_reason
                        }]
                    }
                }
                
                # Add to results (not failed_urls) so it shows as FAIL with proper scoring
                analysis_results[job_id]['results'][url] = {
                    "url": str(url),
                    "channel_id": channel_id,
                    "channel_name": channel_name or "Unknown",
                    "channel_handle": channel_handle or "Unknown",
                    "video_analyses": [controversy_analysis],
                    "summary": controversy_summary,
                    "original_videos": [],
                    "controversy_flagged": True,
                    "controversy_status": controversy_status,
                    "controversy_reason": controversy_reason,
                    "status": "FAIL"  # Explicitly mark as FAIL
                }
                
                logger.info(f"✅ Added controversial channel {url} to results as FAIL: {controversy_reason}")
                
                # Don't process videos for controversial channels
            else:
                logger.debug(f"✅ Controversy Worker {worker_id}: {channel_name} passed controversy screening in {screening_time:.2f}s")
                
                # Add each video to video queue for transcript processing
                for video in videos:
                    # Update pipeline stage: screening -> queued for transcripts
                    update_pipeline_stage(job_id, 'screening_controversy', 'queued_for_transcripts', analysis_results=analysis_results)
                    
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
                        'start_time': time.time(),
                        'controversy_check_failed': controversy_status == 'error',  # Only flag as failed if there was an error
                        'controversy_status': controversy_status
                    })
            
            # Mark task as done
            controversy_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown') if 'item' in locals() else 'unknown'
            channel_name = item.get('channel_name', 'Unknown') if 'item' in locals() else 'Unknown'
            
            logger.error(f"💥 Controversy screening worker {worker_id} error processing {url}: {e}")
            
            # Update pipeline stage: screening -> controversy check failed (but continue)
            update_pipeline_stage(job_id, 'screening_controversy', 'controversy_check_failed', analysis_results=analysis_results)
            
            # Log the failure but continue processing the channel
            logger.warning(f"⚠️ Controversy Worker {worker_id}: Failed to screen {channel_name}, continuing with video processing")
            
            # Add a note to the channel that controversy check failed
            if url not in analysis_results[job_id].get('controversy_check_failures', {}):
                if 'controversy_check_failures' not in analysis_results[job_id]:
                    analysis_results[job_id]['controversy_check_failures'] = {}
                analysis_results[job_id]['controversy_check_failures'][url] = {
                    'channel_name': channel_name,
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Still queue the videos for processing despite controversy check failure
            if 'item' in locals() and 'videos' in item:
                for video in item['videos']:
                    # Update pipeline stage: screening -> queued for transcripts
                    update_pipeline_stage(job_id, 'screening_controversy', 'queued_for_transcripts', analysis_results=analysis_results)
                    
                    # Update video discovery count
                    analysis_results[job_id]['video_progress']['total_videos_discovered'] += 1
                    
                    await video_queue.put({
                        'url': url,
                        'video_id': video['id'],
                        'video_title': video['title'],
                        'video_url': video['url'],
                        'channel_id': item.get('channel_id', 'unknown'),
                        'channel_name': channel_name,
                        'channel_handle': item.get('channel_handle', 'Unknown'),
                        'start_time': time.time(),
                        'controversy_check_failed': True,  # Flag that controversy check failed
                        'controversy_status': 'error'
                    })
            
            controversy_queue.task_done()

async def video_transcript_worker(worker_id: int, video_queue: asyncio.Queue, llm_queue: asyncio.Queue,
                                 youtube_analyzer, timing_stats: dict, job_id: str, analysis_results: dict, result_queue: asyncio.Queue = None):
    """Worker that fetches transcripts for individual videos"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"🎬 Video Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
            
            # Get work item
            item = await video_queue.get()
            
            video_id = item['video_id']
            video_url = item['video_url']
            
            # Track retry attempts
            retry_count = item.get('retry_count', 0)
            max_retries = 3
            
            start_time = time.time()
            
            # Update pipeline stage: queued for transcripts -> fetching transcripts
            update_pipeline_stage(job_id, 'queued_for_transcripts', 'fetching_transcripts', analysis_results=analysis_results)
            
            logger.debug(f"🎬 Video Worker {worker_id}: Processing video {video_id}")
            
            # CHECK CACHE BASED ON ENVIRONMENT SETTINGS
            if video_cache.cache_transcripts_only:
                # TRANSCRIPT-ONLY CACHE MODE: Check for cached transcript only
                cached_transcript = video_cache.get_cached_transcript(video_url)
                if cached_transcript:
                    logger.info(f"🎯 Transcript cache hit for video {video_id}: {item['video_title']}")
                    
                    # Update cache hit counter (but not full cache hit since we still need LLM processing)
                    analysis_results[job_id]['video_progress']['cache_hits'] += 1
                    analysis_results[job_id]['video_progress']['videos_with_transcripts'] += 1
                    
                    # Update pipeline stage: fetching transcripts -> queued for LLM
                    update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_llm', analysis_results=analysis_results)
                    
                    # Pass cached transcript to LLM queue for normal processing
                    await llm_queue.put({
                        'url': item['url'],  # Original channel URL
                        'video_id': video_id,
                        'video_title': item['video_title'],
                        'video_url': item['video_url'],
                        'transcript': cached_transcript['transcript'],
                        'channel_id': item['channel_id'],
                        'channel_name': item['channel_name'],
                        'channel_handle': item['channel_handle'],
                        'start_time': time.time(),
                        'controversy_check_failed': item.get('controversy_check_failed', False),
                        'transcript_cache_hit': True  # Flag to indicate transcript came from cache
                    })
                    
                    video_queue.task_done()
                    continue
            elif not video_cache.cache_disabled:
                # FULL CACHE MODE: Check for full cached result (transcript + LLM)
                cached_result = video_cache.get_cached_result(video_url)
                if cached_result:
                    logger.info(f"🎯 Full cache hit for video {video_id}: {item['video_title']}")
                    
                    # Update cache hit counter
                    analysis_results[job_id]['video_progress']['cache_hits'] += 1
                    analysis_results[job_id]['video_progress']['videos_with_transcripts'] += 1
                    analysis_results[job_id]['video_progress']['videos_analyzed_by_llm'] += 1
                    
                    # Create result entry directly from cache
                    result_item = {
                        'url': item['url'],  # Original channel URL
                        'original_url': item['url'], # Explicitly pass original_url
                        'video_id': video_id,
                        'video_title': item['video_title'],
                        'video_url': video_url,
                        'video_analysis': cached_result['llm_result'],
                        'transcript': cached_result['transcript'],
                        'channel_id': item['channel_id'],
                        'channel_name': item['channel_name'],
                        'channel_handle': item['channel_handle'],
                        'start_time': time.time(),
                        'controversy_check_failed': item.get('controversy_check_failed', False),
                        'cache_hit': True
                    }
                    
                    # Try to use the passed result_queue first, then fallback to queue lookup
                    if result_queue:
                        # Update pipeline stages to reflect cache hit processing
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_llm', analysis_results=analysis_results)
                        update_pipeline_stage(job_id, 'queued_for_llm', 'llm_processing', analysis_results=analysis_results)
                        update_pipeline_stage(job_id, 'llm_processing', 'queued_for_results', analysis_results=analysis_results)
                        
                        await result_queue.put(result_item)
                        logger.debug(f"✅ Cache result queued for video {video_id}")
                    else:
                        # IMPROVED FALLBACK: Process the cache hit directly like the result worker would
                        url = item['url']
                        if url not in analysis_results[job_id]['results']:
                            analysis_results[job_id]['results'][url] = {
                                "url": str(url),
                                "channel_id": item['channel_id'],
                                "channel_name": item['channel_name'],
                                "channel_handle": item['channel_handle'],
                                "video_analyses": [],
                                "summary": {},
                                "original_videos": [],
                                "controversy_flagged": False,
                                "controversy_status": "not_controversial"
                            }
                        
                        # CRITICAL FIX: Add video analysis with CACHED results, not new LLM call
                        video_analysis_entry = {
                            "video_id": video_id,
                            "video_title": item['video_title'],
                            "video_url": video_url,
                            "analysis": cached_result['llm_result'],  # CRITICAL: Use cached LLM result
                            "transcript": cached_result['transcript'],  # CRITICAL: Include transcript
                            "controversy_status": "not_controversial"
                        }
                        
                        analysis_results[job_id]['results'][url]['video_analyses'].append(video_analysis_entry)
                        
                        # CRITICAL FIX: Store original video data for evidence API
                        original_video_data = {
                            "id": video_id,
                            "title": item['video_title'],
                            "url": video_url,
                            "transcript": cached_result['transcript']
                        }
                        analysis_results[job_id]['results'][url]['original_videos'].append(original_video_data)
                        
                        # Update video progress tracking (this was missing!)
                        analysis_results[job_id]['video_progress']['videos_completed'] += 1
                        
                        # Update pipeline stages properly
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_results', analysis_results=analysis_results)
                        update_pipeline_stage(job_id, 'queued_for_results', 'result_processing', analysis_results=analysis_results)
                        update_pipeline_stage(job_id, 'result_processing', 'completed', analysis_results=analysis_results)
                        
                        logger.debug(f"✅ Cache hit processed directly for video {video_id}")
                        
                        # CRITICAL DEBUG: Log what we stored from cache
                        logger.error(f"📦 CACHE HIT DEBUG: Directly stored video analysis for {video_id}")
                        if isinstance(cached_result['llm_result'], dict):
                            if 'error' in cached_result['llm_result']:
                                logger.error(f"   ❌ CACHE HAS ERROR: {cached_result['llm_result'].get('error')}")
                            elif 'results' in cached_result['llm_result']:
                                results = cached_result['llm_result']['results']
                                logger.error(f"   ✅ CACHE HAS {len(results)} categories with scores")
                                for cat, data in list(results.items())[:3]:  # Show first 3
                                    score = data.get('score', 0) if isinstance(data, dict) else 'INVALID'
                                    logger.error(f"      '{cat}': {score}")
                                if len(results) > 3:
                                    logger.error(f"      ... and {len(results) - 3} more categories")
                            else:
                                logger.error(f"   ⚠️ CACHE missing 'results' key! Keys: {list(cached_result['llm_result'].keys())}")
                        else:
                            logger.error(f"   ⚠️ CACHE llm_result is not a dict! Type: {type(cached_result['llm_result'])}")
#                    else:
#                        logger.error(f"❌ No get_queues method available for cached video {video_id}")
                    
                    video_queue.task_done()
                    continue
            # If cache is disabled or no cache hit, proceed with normal transcript fetching
            logger.debug(f"🎬 Video Worker {worker_id}: {'Cache disabled, ' if video_cache.cache_disabled else ''}Fetching transcript for video {video_id}")
            
            # NO CACHE HIT - Proceed with normal transcript fetching
            logger.debug(f"🎬 Video Worker {worker_id}: Fetching transcript for video {video_id}")
            
            # Track the request
            async with youtube_rate_limiter['lock']:
                youtube_rate_limiter['total_transcript_requests'] += 1
            
            # Fetch transcript for this individual video (rate-limited globally)
            transcript = await youtube_analyzer.get_transcript_async(video_id)
            
            transcript_time = time.time() - start_time
            timing_stats['transcript_fetch'].append(transcript_time)
            
            if transcript:
                # Check if we got a rate limit error
                if isinstance(transcript, dict) and transcript.get('error') and 'blocking requests' in transcript.get('error', ''):
                    logger.warning(f"⚠️ YouTube is blocking requests! Initiating backoff...")
                    
                    async with youtube_rate_limiter['lock']:
                        youtube_rate_limiter['consecutive_blocks'] += 1
                        youtube_rate_limiter['last_block_time'] = time.time()
                        
                        # Exponential backoff with jitter
                        backoff = min(
                            youtube_rate_limiter['backoff_seconds'] * (2 ** youtube_rate_limiter['consecutive_blocks']),
                            youtube_rate_limiter['max_backoff_seconds']
                        )
                        youtube_rate_limiter['blocked_until'] = time.time() + backoff
                        
                        logger.warning(f"🛑 YouTube rate limit hit! Backing off for {backoff}s (attempt #{youtube_rate_limiter['consecutive_blocks']})")
                    
                    # Check if we've exceeded max retries
                    if retry_count >= max_retries:
                        logger.error(f"❌ Video Worker {worker_id}: Max retries exceeded for video {video_id}")
                        # Update pipeline stage: fetching transcripts -> failed
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'failed', analysis_results=analysis_results)
                        # Don't add to failed_urls since this is a video-level failure
                        
                        # CRITICAL: Mark the current task as done when giving up
                        video_queue.task_done()
                    else:
                        # Put the item back in the queue for retry with incremented count
                        item['retry_count'] = retry_count + 1
                        await video_queue.put(item)
                        
                        # Update pipeline stage back to queued for retry
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_transcripts', analysis_results=analysis_results)
                        
                        logger.info(f"🔄 Video Worker {worker_id}: Retrying video {video_id} (attempt {retry_count + 1}/{max_retries})")
                        
                        # CRITICAL: Mark the current task as done before retrying
                        video_queue.task_done()
                else:
                    # Success! Reset consecutive blocks
                    async with youtube_rate_limiter['lock']:
                        youtube_rate_limiter['consecutive_blocks'] = 0
                    
                    logger.debug(f"✅ Video Worker {worker_id}: Got transcript for {video_id} in {transcript_time:.2f}s")
                    
                    # Update video progress tracking
                    analysis_results[job_id]['video_progress']['videos_with_transcripts'] += 1
                    
                    # Update pipeline stage: fetching transcripts -> queued for LLM
                    update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_llm', analysis_results=analysis_results)
                    
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
                        'start_time': time.time(),
                        'controversy_check_failed': item.get('controversy_check_failed', False)  # Pass the flag
                    })
                    
                    # Mark task as done for successful transcript fetch
                    video_queue.task_done()
            else:
                # Hide verbose logging for common transcript failures
                logger.debug(f"❌ Video Worker {worker_id}: No transcript available for video {video_id}")
                # Update pipeline stage: fetching transcripts -> failed (for this video)
                update_pipeline_stage(job_id, 'fetching_transcripts', 'failed', analysis_results=analysis_results)
                # Don't count this as a failure since we got some videos from the channel
                
                video_queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            video_id = item.get('video_id', 'unknown') if 'item' in locals() else 'unknown'
            logger.error(f"💥 Video transcript worker {worker_id} error processing {video_id}: {e}")
            video_queue.task_done()

async def llm_worker(worker_id: int, llm_queue: asyncio.Queue, result_queue: asyncio.Queue,
                     llm_analyzer, timing_stats: dict, job_id: str, analysis_results: dict):
    """Worker that processes individual videos through LLM"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"🤖 LLM Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
                
            # Get work item
            item = await llm_queue.get()
            url = item['url']  # Original channel URL
            video_id = item['video_id']
            transcript = item['transcript']
            start_time = time.time()
            
            # Update pipeline stage: queued for LLM -> LLM processing
            update_pipeline_stage(job_id, 'queued_for_llm', 'llm_processing', analysis_results=analysis_results)
            
            logger.debug(f"🤖 LLM Worker {worker_id}: Processing video {video_id}")
            
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
            timing_stats['llm_analysis'].append(llm_time)
            
            # Update video progress tracking
            analysis_results[job_id]['video_progress']['videos_analyzed_by_llm'] += 1
            
            logger.debug(f"✅ LLM Worker {worker_id}: Completed video {video_id} in {llm_time:.2f}s")
            
            # Update pipeline stage: LLM processing -> queued for results
            update_pipeline_stage(job_id, 'llm_processing', 'queued_for_results', analysis_results=analysis_results)
            
            # Pass to result queue with all original channel info
            await result_queue.put({
                'url': url,  # Original channel URL (this is the key for the results dict)
                'original_url': url, # Explicitly pass original_url for the result worker
                'video_id': video_id,
                'video_title': item['video_title'],
                'video_url': item['video_url'],
                'video_analysis': analysis_result,
                'transcript': transcript,
                'channel_id': item['channel_id'],
                'channel_name': item['channel_name'],
                'channel_handle': item['channel_handle'],
                'start_time': time.time(),
                'controversy_check_failed': item.get('controversy_check_failed', False)  # Pass the flag
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
            update_pipeline_stage(job_id, 'llm_processing', 'failed', analysis_results=analysis_results)
            
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'llm_processing_failed',
                'channel_name': channel_name,
                'video_id': video_id
            })
            llm_queue.task_done()

async def result_worker(worker_id: int, result_queue: asyncio.Queue, timing_stats: dict, job_id: str, analysis_results: dict):
    """Worker that aggregates video results by channel and stores them"""
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"📊 Result Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
            
            # Get work item
            item = await result_queue.get()
            url = item['url']  # Original channel URL
            video_id = item['video_id']
            video_title = item['video_title']
            video_url = item['video_url']
            video_analysis = item['video_analysis']
            transcript = item['transcript']
            channel_id = item['channel_id']
            # Use .get for channel_name and channel_handle for safety, though they should be there
            raw_channel_name = item.get('channel_name', '') 
            raw_channel_handle = item.get('channel_handle', '')
            controversy_status = item.get('controversy_status', 'not_controversial')
            
            # Get or create channel entry
            if url not in analysis_results[job_id]['results']:
                retrieved_channel_id = item.get('channel_id', 'unknown') # Use a consistent var name

                # Clean name and handle
                final_channel_name = raw_channel_name.strip() if isinstance(raw_channel_name, str) else ''
                final_channel_handle = raw_channel_handle.strip() if isinstance(raw_channel_handle, str) else ''
                
                if not final_channel_name and retrieved_channel_id != "unknown": # Check against "unknown"
                    final_channel_name = f"Channel {retrieved_channel_id}" # Fallback name

                # If handle is empty/None, but name contains something like "(@handle)", try to extract it.
                if not final_channel_handle and final_channel_name:
                    # Regex to find @handle, possibly in parentheses, and ensure it's a valid-looking handle
                    handle_match = re.search(r'\(?(@[a-zA-Z0-9_.-]+)\)?', final_channel_name)
                    if handle_match:
                        potential_handle = handle_match.group(1)
                        # Optional: Consider if cleaning the name is desired here
                        # final_channel_name = final_channel_name.replace(handle_match.group(0), '').strip()
                        final_channel_handle = potential_handle
                        logger.info(f"🔧 RESULT_WORKER: Extracted potential handle '{final_channel_handle}' from channel name '{raw_channel_name}' for {url}")
                
                # Ensure channel_name and channel_handle from item are used if already good
                # The logic above refines them, so we use final_channel_name and final_channel_handle

                controversy_info = None
                if url in analysis_results[job_id].get('controversy_check_failures', {}):
                    controversy_info = analysis_results[job_id]['controversy_check_failures'][url]
                
                # CRITICAL FIX: Remove from failed_urls if it was prematurely added
                # This prevents double-counting when a channel was marked as "missing" but then completes successfully
                failed_urls = analysis_results[job_id]['failed_urls']
                failed_urls_before = len(failed_urls)
                analysis_results[job_id]['failed_urls'] = [f for f in failed_urls if f['url'] != url]
                failed_urls_after = len(analysis_results[job_id]['failed_urls'])
                
                if failed_urls_after < failed_urls_before:
                    logger.info(f"🔧 Removed {url} from failed_urls (was prematurely marked as failed) - preventing double-counting")
                
                # Initialize with empty lists for video_analyses and original_videos
                video_analyses_for_channel = [] # Ensure this is initialized
                summary_for_channel = {} # Ensure this is initialized
                original_videos_for_channel = [] # Ensure this is initialized

                # Prepare the final result data for this channel
                result_data_for_channel = {
                    'url': str(url),  # Ensure this is the original channel URL, not "unknown"
                    'channel_id': retrieved_channel_id,
                    'channel_name': final_channel_name or "Unknown",
                    'channel_handle': final_channel_handle or "Unknown",
                    'original_input_url': url,  # Store original URL for reference
                    'video_analyses': video_analyses_for_channel, # Use initialized list
                    'summary': summary_for_channel, # Use initialized dict
                    'original_videos': original_videos_for_channel, # Use initialized list
                    # Populate controversy_check_result consistently
                    'controversy_check_result': {
                        'status': item.get('controversy_status', 'not_screened'),
                        'reason': item.get('controversy_reason', '') # This might be empty if not controversial/error
                    }
                }
                
                analysis_results[job_id]['results'][url] = result_data_for_channel
            
            # Add this video's analysis to the channel
            video_analysis_entry = {
                "video_id": item['video_id'],
                "video_title": item['video_title'],
                "video_url": item['video_url'],
                "analysis": item['video_analysis'],
                "transcript": transcript,  # CRITICAL FIX: Include transcript in video analysis entry
                "controversy_status": controversy_status
            }
            
            # CRITICAL DEBUG: Log what we're actually storing
            logger.error(f"📊 RESULT WORKER DEBUG: Storing video analysis for {video_id}")
            if isinstance(item['video_analysis'], dict):
                if 'error' in item['video_analysis']:
                    logger.error(f"   ❌ Video has error: {item['video_analysis'].get('error')}")
                elif 'results' in item['video_analysis']:
                    results = item['video_analysis']['results']
                    logger.error(f"   ✅ Video has {len(results)} categories with scores")
                    for cat, data in list(results.items())[:3]:  # Show first 3
                        score = data.get('score', 0) if isinstance(data, dict) else 'INVALID'
                        logger.error(f"      '{cat}': {score}")
                    if len(results) > 3:
                        logger.error(f"      ... and {len(results) - 3} more categories")
                else:
                    logger.error(f"   ⚠️ Video analysis missing 'results' key! Keys: {list(item['video_analysis'].keys())}")
            else:
                logger.error(f"   ⚠️ Video analysis is not a dict! Type: {type(item['video_analysis'])}")
            
            analysis_results[job_id]['results'][url]['video_analyses'].append(video_analysis_entry)
            
            # CRITICAL FIX: Store original video data for evidence API
            original_video_data = {
                "id": item['video_id'],
                "title": item['video_title'],
                "url": item['video_url'],
                "transcript": {
                    "full_text": transcript
                } if transcript else None
            }
            # Ensure 'original_videos' list exists before appending
            if 'original_videos' not in analysis_results[job_id]['results'][url]:
                analysis_results[job_id]['results'][url]['original_videos'] = []
            analysis_results[job_id]['results'][url]['original_videos'].append(original_video_data)
            
            # RECALCULATE OVERALL SCORE FOR THE CHANNEL after adding this video analysis
            current_max_overall_score_for_channel = 0.0
            
            # Use the imported all_categories_glob for category matching
            if not all_categories_glob:
                logger.error(f"RESULT_WORKER: Global category list (all_categories_glob) is empty. Cannot accurately calculate overall score for channel {url}.")
            
            for vid_analysis in analysis_results[job_id]['results'][url]['video_analyses']:
                analysis_content = vid_analysis.get('analysis', {})
                llm_results_for_this_video = analysis_content.get('results', {})
                for llm_cat_name, llm_cat_data in llm_results_for_this_video.items():
                    score = llm_cat_data.get('score', 0.0)
                    if not isinstance(score, (int, float)) or float(score) <= 0:
                        continue
                    
                    normalized_llm_cat_name = llm_cat_name.lower().strip()
                    is_recognized = any(normalized_llm_cat_name == defined_cat.lower().strip() or \
                                        normalized_llm_cat_name in defined_cat.lower().strip() or \
                                        defined_cat.lower().strip() in normalized_llm_cat_name 
                                        for defined_cat in all_categories_glob) # Use imported list

                    if is_recognized:
                        current_max_overall_score_for_channel = max(current_max_overall_score_for_channel, float(score))
            
            # Update the summary with the new overall_score
            if 'summary' not in analysis_results[job_id]['results'][url]:
                analysis_results[job_id]['results'][url]['summary'] = {}
            analysis_results[job_id]['results'][url]['summary']['overall_score'] = round(current_max_overall_score_for_channel, 2)
            
            # ALSO store overall_score at the top level for UI compatibility
            analysis_results[job_id]['results'][url]['overall_score'] = round(current_max_overall_score_for_channel, 2)
            
            logger.info(f"RESULT_WORKER: Updated overall_score for channel {url} to {analysis_results[job_id]['results'][url]['summary']['overall_score']}")

            # Update video progress tracking
            analysis_results[job_id]['video_progress']['videos_completed'] += 1
            
            # Update pipeline stage: queued for results -> result processing
            update_pipeline_stage(job_id, 'queued_for_results', 'result_processing', analysis_results=analysis_results)
            
            # Store result in cache for future use (unless it's already a cache hit)
            if not item.get('cache_hit', False) and not video_cache.cache_disabled:
                # Store in cache unless cache is completely disabled
                cache_stored = video_cache.store_result(
                    video_url=video_url,
                    video_id=video_id,
                    video_title=video_title,
                    channel_id=channel_id,
                    channel_name=final_channel_name,
                    transcript=transcript,
                    llm_result=video_analysis
                )
                if cache_stored:
                    cache_mode = "transcript-only" if video_cache.cache_transcripts_only else "full"
                    logger.debug(f"💾 Stored video {video_id} in cache ({cache_mode} mode)")
                else:
                    logger.warning(f"⚠️ Failed to store video {video_id} in cache")
            elif item.get('transcript_cache_hit', False):
                # This was a transcript cache hit, so we should store the LLM result to complete the cache entry
                if not video_cache.cache_disabled:
                    cache_stored = video_cache.store_result(
                        video_url=video_url,
                        video_id=video_id,
                        video_title=video_title,
                        channel_id=channel_id,
                        channel_name=final_channel_name,
                        transcript=transcript,
                        llm_result=video_analysis
                    )
                    if cache_stored:
                        logger.debug(f"💾 Updated cache entry for video {video_id} with LLM result")
                    else:
                        logger.warning(f"⚠️ Failed to update cache entry for video {video_id}")
            elif video_cache.cache_disabled:
                logger.debug(f"📦 Cache disabled - not storing video {video_id}")
            else:
                logger.debug(f"📦 Video {video_id} already from cache - not storing again")
            
            # Mark task as done
            result_queue.task_done()
            
            # After successfully processing the result, move from result_processing to completed
            update_pipeline_stage(job_id, 'result_processing', 'completed', analysis_results=analysis_results)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            url = item.get('url', 'unknown') if 'item' in locals() else 'unknown'
            video_id = item.get('video_id', 'unknown')
            channel_name = item.get('channel_name', 'Unknown')
            error_msg = f"Failed to process results for video {video_id} from channel {channel_name}: {str(e)}"
            
            logger.error(f"💥 Result worker {worker_id} error processing video {video_id}: {e}")
            
            # Update pipeline stage: result processing -> failed
            update_pipeline_stage(job_id, 'result_processing', 'failed', analysis_results=analysis_results)
            
            # Add to failed URLs with clear messaging
            analysis_results[job_id]['failed_urls'].append({
                'url': url,
                'error': error_msg,
                'error_type': 'result_processing_failed',
                'channel_name': channel_name,
                'video_id': video_id
            })
            
            result_queue.task_done()

async def monitor_pipeline_detailed(job_id: str, channel_queue: asyncio.Queue = None, controversy_queue: asyncio.Queue = None, 
                                   video_queue: asyncio.Queue = None, llm_queue: asyncio.Queue = None, 
                                   result_queue: asyncio.Queue = None, total_urls: int = None, 
                                   timing_stats: dict = None, analysis_results: dict = None):
    """
    Monitor pipeline progress and update job status with detailed metrics.
    Can be used in two modes:
    1. Full monitoring mode: When all parameters are provided, monitors queue depths and progress
    2. Status update mode: When only job_id is provided, just updates the job's ETA and progress stats
    """
    try:
        job = analysis_results.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found in analysis_results")
            return

        # Calculate ETA using shared function
        eta_info = calculate_job_eta(job)
        
        # Update job with ETA info
        job['performance_stats']['eta_info'] = eta_info
        
        # Update processing rate in performance stats
        if eta_info.get('processing_rate_per_minute'):
            job['performance_stats']['processing_rate_per_minute'] = eta_info['processing_rate_per_minute']
        
        # Log progress
        logger.info(f"📊 Job {job_id} progress update:")
        logger.info(f"   📈 Channel progress: {len(job['results'])}/{job['total_urls']} channels processed")
        logger.info(f"   📊 Video progress: {job.get('video_progress', {}).get('videos_completed', 0)}/{job.get('video_progress', {}).get('total_videos_discovered', 0)} videos completed")
        logger.info(f"   🎯 Cache hits: {job.get('video_progress', {}).get('cache_hits', 0)} videos from cache")
        logger.info(f"   ⏱️ ETA: {eta_info.get('estimated_minutes_remaining', 'N/A')} minutes remaining")

        # If we're in full monitoring mode (all parameters provided)
        if all([channel_queue, controversy_queue, video_queue, llm_queue, result_queue, total_urls, timing_stats]):
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
                    controversy_size = controversy_queue.qsize()
                    video_size = video_queue.qsize() 
                    llm_size = llm_queue.qsize() 
                    result_size = result_queue.qsize()
                    
                    # REVERTED PROGRESS CALCULATION: Use URL-based completion
                    # Calculate completion based on processed URLs
                    url_completed = len(analysis_results[job_id]['results'])
                    url_failed = len(analysis_results[job_id]['failed_urls'])
                    url_processed = url_completed + url_failed
                    progress_pct = (url_processed / total_urls * 100) if total_urls > 0 else 0
                    progress_display = f"{url_processed}/{total_urls} URLs"
                    
                    logger.info(f"📊 D:{channel_size:2d} | C:{controversy_size:2d} | T:{video_size:2d} | L:{llm_size:2d} | R:{result_size:2d} | {progress_display} ({progress_pct:5.1f}%) | {elapsed:6.1f}s elapsed")
                    
                    # Update queue depths for analysis
                    if 'queue_depths' not in timing_stats:
                        timing_stats['queue_depths'] = {
                            'channel': [],
                            'controversy': [],
                            'video': [],
                            'llm': [],
                            'result': []
                        }
                        timing_stats['timestamps'] = []
                    
                    timing_stats['queue_depths']['channel'].append(channel_size)
                    timing_stats['queue_depths']['controversy'].append(controversy_size)
                    timing_stats['queue_depths']['video'].append(video_size)
                    timing_stats['queue_depths']['llm'].append(llm_size)
                    timing_stats['queue_depths']['result'].append(result_size)
                    timing_stats['timestamps'].append(current_time)
                    
                    # Detailed logging every 15 seconds
                    if elapsed - last_detailed_log >= 15:
                        last_detailed_log = elapsed
                        
                        # Calculate processing rate based on total work completed
                        if url_processed > 0:
                            rate = url_processed / elapsed
                            eta_info = calculate_job_eta(analysis_results[job_id])
                            
                            # Debug current max values
                            current_max_c = max(timing_stats['queue_depths']['channel'], default=0)
                            current_max_cont = max(timing_stats['queue_depths']['controversy'], default=0)
                            current_max_v = max(timing_stats['queue_depths']['video'], default=0)
                            current_max_l = max(timing_stats['queue_depths']['llm'], default=0)
                            current_max_r = max(timing_stats['queue_depths']['result'], default=0)
                            
                            logger.info(f"📈 DETAILED STATUS:")
                            logger.info(f"   └─ Processing rate: {rate:.2f} items/sec")
                            logger.info(f"   └─ ETA: {eta_info.get('estimated_minutes_remaining', 'N/A')} minutes")
                            logger.info(f"   └─ Queue depths - Max seen: D:{current_max_c} | C:{current_max_cont} | T:{current_max_v} | L:{current_max_l} | R:{current_max_r}")
                            logger.info(f"   └─ Pipeline stages: {analysis_results[job_id]['pipeline_stages']}")
                    
                    # If all work is done, break - IMPROVED COMPLETION CHECK
                    work_complete = False
                    if url_processed >= total_urls and channel_size == 0 and controversy_size == 0 and video_size == 0 and llm_size == 0 and result_size == 0:
                        logger.info("🏁 All work completed, stopping monitor")
                        logger.info(f"📊 Final monitoring stats: {len(timing_stats['queue_depths']['channel'])} data points collected over {elapsed:.1f}s")
                        break
                    
                    # Deadlock detection: if we have items in transcript queue but no progress for a long time
                    if video_size > 0 and elapsed > 300:  # 5 minutes
                        # Check if we've made progress in the last 2 minutes
                        recent_progress = False
                        if len(timing_stats['timestamps']) >= 40:  # 40 * 3 seconds = 2 minutes
                            old_completed = 0
                            # Look at progress from 2 minutes ago
                            for i, timestamp in enumerate(timing_stats['timestamps'][-40:]):
                                if timestamp <= current_time - 120:  # 2 minutes ago
                                    # Count completed work at that time
                                    old_completed = url_processed
                                    break
                            
                            if url_processed > old_completed:
                                recent_progress = True
                        
                        if not recent_progress and video_size > 0:
                            logger.warning(f"🚨 POTENTIAL DEADLOCK DETECTED:")
                            logger.warning(f"   └─ {video_size} items stuck in transcript queue for >2 minutes")
                            logger.warning(f"   └─ YouTube rate limiter status: blocked_until={youtube_rate_limiter.get('blocked_until')}")
                            logger.warning(f"   └─ Current time: {time.time()}")
                            if youtube_rate_limiter.get('blocked_until'):
                                remaining_block = youtube_rate_limiter['blocked_until'] - time.time()
                                logger.warning(f"   └─ Remaining block time: {remaining_block:.1f}s")
                        
                except asyncio.CancelledError:
                    logger.info("🔍 Pipeline monitor cancelled")
                    break
                except Exception as e:
                    logger.error(f"💥 Monitor error: {e}")
                    logger.error(traceback.format_exc())
                    break
        
    except Exception as e:
        logger.error(f"Error in monitor_pipeline_detailed: {str(e)}")
        logger.error(traceback.format_exc())

async def process_job_with_cleanup(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Process a job with proper cleanup and queue management"""
    try:
        # Create queues
        channel_queue = asyncio.Queue()
        controversy_queue = asyncio.Queue()
        video_queue = asyncio.Queue()
        llm_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        
        # Store queues for status updates
        queues = {
            'channel_queue': channel_queue,
            'controversy_queue': controversy_queue,
            'video_queue': video_queue,
            'transcript_queue': video_queue,  # Alias for status endpoint
            'llm_queue': llm_queue
        }
        
        # Initialize YouTube API call counter for this job
        youtube_rate_limiter['total_api_calls'] = 0
        logger.info(f"🔄 Reset YouTube API call counter to 0 for job {job_id}")
        
        # Initialize analyzers
        youtube_analyzer = YouTubeAnalyzer(youtube_api_key=os.getenv("YOUTUBE_API_KEY"))
        llm_analyzer = LLMAnalyzer(provider=llm_provider)
        
        # Initialize timing stats
        timing_stats = {
            'channel_discovery': [],
            'controversy_screening': [],
            'transcript_fetch': [],
            'llm_analysis': []
        }
        
        # Create worker tasks
        workers = []
        
        # Channel discovery workers
        for i in range(3):  # 3 channel discovery workers
            worker = asyncio.create_task(
                channel_discovery_worker(i, channel_queue, controversy_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = lambda: queues
            worker.clear_queues = lambda: clear_queues(queues)
            workers.append(worker)
        
        # Controversy screening workers
        for i in range(2):  # 2 controversy screening workers
            worker = asyncio.create_task(
                controversy_screening_worker(i, controversy_queue, video_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = lambda: queues
            worker.clear_queues = lambda: clear_queues(queues)
            workers.append(worker)
        
        # Video transcript workers
        for i in range(5):  # 5 video transcript workers
            worker = asyncio.create_task(
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id, analysis_results, result_queue)
            )
            worker.get_queues = lambda: queues
            worker.clear_queues = lambda: clear_queues(queues)
            workers.append(worker)
        
        # LLM workers
        for i in range(MAX_LLM_WORKERS):  # Use the constant
            worker = asyncio.create_task(
                llm_worker(i, llm_queue, result_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = lambda: queues
            worker.clear_queues = lambda: clear_queues(queues)
            workers.append(worker)
        
        # Result worker
        result_worker_task = asyncio.create_task(
            result_worker(0, result_queue, timing_stats, job_id, analysis_results)
        )
        result_worker_task.get_queues = lambda: queues
        result_worker_task.clear_queues = lambda: clear_queues(queues)
        workers.append(result_worker_task)
        
        # Monitor task
        monitor_task = asyncio.create_task(
            monitor_pipeline_detailed(
                job_id, channel_queue, controversy_queue, video_queue, 
                llm_queue, result_queue, len(urls), timing_stats, analysis_results
            )
        )
        monitor_task.get_queues = lambda: queues
        monitor_task.clear_queues = lambda: clear_queues(queues)
        workers.append(monitor_task)
        
        # Add all workers to active_job_tasks if it exists
        if 'active_job_tasks' in globals():
            active_job_tasks[job_id] = workers
        
        # Add URLs to channel queue
        for url in urls:
            await channel_queue.put({
                'url': url,
                'video_limit': video_limit,
                'start_time': time.time()
            })
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
        
        # Mark job as completed
        analysis_results[job_id]['status'] = 'completed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        
        # Store final YouTube API call count for this job
        if 'performance_stats' in analysis_results[job_id]:
            analysis_results[job_id]['performance_stats']['youtube_api_calls_for_job'] = youtube_rate_limiter['total_api_calls']
            logger.info(f"📊 Job {job_id} completed. Total YouTube API calls for this job: {youtube_rate_limiter['total_api_calls']}")
        else:
            logger.warning(f"⚠️ Could not store YouTube API call count for job {job_id} - performance_stats missing.")

        logger.info(f"✅ Job {job_id} completed successfully.")
        
        # Try to start the next job in the queue
        job_manager.current_job_id = None
        job_completion_events[job_id].set() # Signal completion for monitor
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mark job as failed
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['error'] = str(e)
    finally:
        # Clean up
        if 'active_job_tasks' in globals() and job_id in active_job_tasks:
            del active_job_tasks[job_id]

async def clear_queues(queues: dict):
    """Clear all queues to prevent new work from being picked up"""
    for queue_name, queue in queues.items():
        items_removed = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                items_removed += 1
            except asyncio.QueueEmpty:
                break
        
        # Only call task_done() for items we actually removed
        for _ in range(items_removed):
            try:
                queue.task_done()
            except ValueError:
                # If we get "task_done() called too many times", stop calling it
                break
        
        if items_removed > 0:
            logger.debug(f"Cleared {items_removed} items from {queue_name}") 