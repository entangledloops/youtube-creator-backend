"""
Pipeline worker functions for processing YouTube channels through various stages
"""
import asyncio
import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import traceback

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.rate_limiter import youtube_rate_limiter
from src.controversy_screener import screen_creator_for_controversy

logger = logging.getLogger(__name__)

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
            
            # Get channel info and video list (without transcripts yet)
            channel_id, channel_name, channel_handle = youtube_analyzer.extract_channel_info_from_url(url)
            
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
                # Mark task as done
                channel_queue.task_done()
                logger.debug(f"📋 Channel Worker {worker_id}: Marked invalid channel as done (queue size: {channel_queue.qsize()})")
            else:
                # Get video list from channel
                videos = youtube_analyzer.get_videos_from_channel(channel_id, limit=video_limit)
                
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
                llm_analyzer
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
                
                # Still queue the videos for processing despite controversy
                for video in videos:
                    # Update pipeline stage: screening -> queued for transcripts
                    update_pipeline_stage(job_id, 'screening_controversy', 'queued_for_transcripts', analysis_results=analysis_results)
                    
                    # Update video discovery count
                    analysis_results[job_id]['video_progress']['total_videos_discovered'] += 1
                    
                    await video_queue.put({
                        'url': url,
                        'video_id': video['id'],
                        'video_title': video['title'],
                        'video_url': video['url'],
                        'channel_id': channel_id,
                        'channel_name': channel_name,
                        'channel_handle': channel_handle,
                        'start_time': time.time(),
                        'controversy_check_failed': True,  # Flag that controversy check failed
                        'controversy_status': controversy_status
                    })
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
                                 youtube_analyzer, timing_stats: dict, job_id: str, analysis_results: dict):
    """Worker that fetches individual video transcripts with global rate limiting"""
    import random
    
    while True:
        try:
            # Check if job is cancelled
            if is_job_cancelled(job_id, analysis_results):
                logger.info(f"🎬 Video Worker {worker_id}: Job {job_id} cancelled, stopping...")
                break
            
            # Check if we're in backoff period
            async with youtube_rate_limiter['lock']:
                if youtube_rate_limiter['blocked_until']:
                    wait_time = youtube_rate_limiter['blocked_until'] - time.time()
                    if wait_time > 0:
                        # Add random jitter to prevent thundering herd
                        jitter = random.uniform(0, min(10, wait_time * 0.1))
                        logger.debug(f"🎬 Video Worker {worker_id}: In backoff period, waiting {wait_time + jitter:.1f}s")
                        await asyncio.sleep(wait_time + jitter)
                        continue
                
            # Get work item
            item = await video_queue.get()
            video_id = item['video_id']
            
            # Track retry attempts
            retry_count = item.get('retry_count', 0)
            max_retries = 3
            
            start_time = time.time()
            
            # Update pipeline stage: queued for transcripts -> fetching transcripts
            update_pipeline_stage(job_id, 'queued_for_transcripts', 'fetching_transcripts', analysis_results=analysis_results)
            
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
                    else:
                        # Put the item back in the queue for retry with incremented count
                        item['retry_count'] = retry_count + 1
                        await video_queue.put(item)
                        
                        # Update pipeline stage back
                        update_pipeline_stage(job_id, 'fetching_transcripts', 'queued_for_transcripts', analysis_results=analysis_results)
                    
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
            else:
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
                'url': url,  # Original channel URL
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
            channel_name = item['channel_name']
            channel_handle = item['channel_handle']
            controversy_status = item.get('controversy_status', 'not_controversial')
            
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
                
                # Check if this channel was flagged for controversy
                controversy_info = None
                if url in analysis_results[job_id].get('controversy_check_failures', {}):
                    controversy_info = analysis_results[job_id]['controversy_check_failures'][url]
                
                analysis_results[job_id]['results'][url] = {
                    "url": str(url),
                    "channel_id": item.get('channel_id', "unknown"),
                    "channel_name": channel_name or "Unknown",
                    "channel_handle": channel_handle or "Unknown", 
                    "video_analyses": [],
                    "summary": {},
                    "original_videos": [],
                    "controversy_flagged": controversy_status == 'controversial',
                    "controversy_status": controversy_status,
                    "controversy_reason": controversy_info.get('reason') if controversy_info else None
                }
            
            # Add this video's analysis to the channel
            video_analysis_entry = {
                "video_id": item['video_id'],
                "video_title": item['video_title'],
                "video_url": item['video_url'],
                "analysis": item['video_analysis'],
                "controversy_status": controversy_status
            }
            
            analysis_results[job_id]['results'][url]['video_analyses'].append(video_analysis_entry)
            
            # Update video progress tracking
            analysis_results[job_id]['video_progress']['videos_completed'] += 1
            
            # Update pipeline stage: queued for results -> result processing
            update_pipeline_stage(job_id, 'queued_for_results', 'result_processing', analysis_results=analysis_results)
            
            # Mark task as done
            result_queue.task_done()
            
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
                    
                    completed = len(analysis_results[job_id]['results'])
                    failed = len(analysis_results[job_id]['failed_urls'])
                    processed = completed + failed
                    
                    # Always log basic status
                    progress_pct = (processed/total_urls*100) if total_urls > 0 else 0
                    logger.info(f"📊 D:{channel_size:2d} | C:{controversy_size:2d} | T:{video_size:2d} | L:{llm_size:2d} | R:{result_size:2d} | {processed:3d}/{total_urls} ({progress_pct:5.1f}%) | {elapsed:6.1f}s elapsed")
                    
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
                        
                        if processed > 0:
                            rate = processed / elapsed
                            eta_info = calculate_job_eta(analysis_results[job_id])
                            
                            # Debug current max values
                            current_max_c = max(timing_stats['queue_depths']['channel'], default=0)
                            current_max_cont = max(timing_stats['queue_depths']['controversy'], default=0)
                            current_max_v = max(timing_stats['queue_depths']['video'], default=0)
                            current_max_l = max(timing_stats['queue_depths']['llm'], default=0)
                            current_max_r = max(timing_stats['queue_depths']['result'], default=0)
                            
                            logger.info(f"📈 DETAILED STATUS:")
                            logger.info(f"   └─ Processing rate: {rate:.2f} URLs/sec")
                            logger.info(f"   └─ ETA: {eta_info.get('estimated_minutes_remaining', 'N/A')} minutes")
                            logger.info(f"   └─ Queue depths - Max seen: D:{current_max_c} | C:{current_max_cont} | T:{current_max_v} | L:{current_max_l} | R:{current_max_r}")
                            logger.info(f"   └─ Pipeline stages: {analysis_results[job_id]['pipeline_stages']}")
                    
                    # If all work is done, break
                    if processed >= total_urls and channel_size == 0 and controversy_size == 0 and video_size == 0 and llm_size == 0 and result_size == 0:
                        logger.info("🏁 All work completed, stopping monitor")
                        logger.info(f"📊 Final monitoring stats: {len(timing_stats['queue_depths']['channel'])} data points collected over {elapsed:.1f}s")
                        break
                        
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
        
        # Initialize analyzers
        youtube_analyzer = YouTubeAnalyzer()
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
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = lambda: queues
            worker.clear_queues = lambda: clear_queues(queues)
            workers.append(worker)
        
        # LLM workers
        for i in range(2):  # 2 LLM workers
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
        
        # Add all workers to active_job_tasks
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
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mark job as failed
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['error'] = str(e)
    finally:
        # Clean up
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]

async def clear_queues(queues: dict):
    """Clear all queues to prevent new work from being picked up"""
    for queue in queues.values():
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break 