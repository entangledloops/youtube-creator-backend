"""
Job queue management and processing for bulk analysis
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import os
import traceback

from src.youtube_analyzer import YouTubeAnalyzer
from src.llm_analyzer import LLMAnalyzer
from src.pipeline_workers import (
    channel_discovery_worker, controversy_screening_worker,
    video_transcript_worker, llm_worker, result_worker,
    monitor_pipeline_detailed, update_pipeline_stage,
    clear_queues
)

logger = logging.getLogger(__name__)

# Global job queue for sequential processing
job_queue = []
current_job_id = None
job_queue_lock = asyncio.Lock()

# Global task tracking for cancellation
active_job_tasks = {}  # job_id -> list of asyncio tasks

async def process_creators_pipeline(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Queue-based pipeline with proper video-level rate limiting - NON-BLOCKING"""
    try:
        logger.info(f"🚀 Starting pipeline for job {job_id} with {len(urls)} URLs")
        
        # Update job status
        analysis_results[job_id]['status'] = 'processing'
        analysis_results[job_id]['started_at'] = datetime.now().isoformat()
        
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
            'llm_queue': llm_queue,
            'result_queue': result_queue
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
        for i in range(3):
            worker = asyncio.create_task(
                channel_discovery_worker(i, channel_queue, controversy_queue, youtube_analyzer, timing_stats, job_id, analysis_results)
            )
            # Create closure to properly capture queues
            def make_get_queues(q):
                return lambda: q
            def make_clear_queues(q):
                return lambda: clear_queues(q)
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Controversy screening workers
        for i in range(2):
            worker = asyncio.create_task(
                controversy_screening_worker(i, controversy_queue, video_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Video transcript workers
        for i in range(3):  # Increased from 2 to 3 per user request
            worker = asyncio.create_task(
                video_transcript_worker(i, video_queue, llm_queue, youtube_analyzer, timing_stats, job_id, analysis_results, result_queue)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # LLM workers
        for i in range(2):
            worker = asyncio.create_task(
                llm_worker(i, llm_queue, result_queue, llm_analyzer, timing_stats, job_id, analysis_results)
            )
            worker.get_queues = make_get_queues(queues)
            worker.clear_queues = make_clear_queues(queues)
            workers.append(worker)
        
        # Result worker
        result_worker_task = asyncio.create_task(
            result_worker(0, result_queue, timing_stats, job_id, analysis_results)
        )
        result_worker_task.get_queues = make_get_queues(queues)
        result_worker_task.clear_queues = make_clear_queues(queues)
        workers.append(result_worker_task)
        
        # Monitor task
        monitor_task = asyncio.create_task(
            monitor_pipeline_detailed(
                job_id, channel_queue, controversy_queue, video_queue, 
                llm_queue, result_queue, len(urls), timing_stats, analysis_results
            )
        )
        monitor_task.get_queues = make_get_queues(queues)
        monitor_task.clear_queues = make_clear_queues(queues)
        workers.append(monitor_task)
        
        # Create a task to add URLs to the queue asynchronously
        async def add_urls_to_queue():
            """Add URLs to channel queue asynchronously"""
            try:
                logger.info(f"📊 Starting to queue {len(urls)} channels for processing")
                for i, url in enumerate(urls):
                    # Update pipeline stage to show URLs being queued
                    update_pipeline_stage(job_id, None, 'queued_for_discovery', analysis_results=analysis_results)
                    
                    await channel_queue.put({
                        'url': url,
                        'video_limit': video_limit,
                        'start_time': time.time()
                    })
                    
                    # Log progress every 10 URLs
                    if (i + 1) % 10 == 0:
                        logger.debug(f"📊 Queued {i + 1}/{len(urls)} URLs for discovery")
                        
                logger.info(f"✅ Finished queuing all {len(urls)} URLs for job {job_id}")
            except Exception as e:
                logger.error(f"Error adding URLs to queue: {str(e)}")
        
        # Start the URL queuing task - NON-BLOCKING
        url_task = asyncio.create_task(add_urls_to_queue())
        workers.append(url_task)  # Add to workers list so it gets cancelled if job is cancelled
        
        # Completion monitor task - this will handle job completion
        # CRITICAL: Pass url_task so monitor can wait for all URLs to be queued
        completion_task = asyncio.create_task(
            monitor_job_completion(job_id, urls, workers, queues, analysis_results, url_task)
        )
        workers.append(completion_task)
        
        # Store all workers for cancellation
        active_job_tasks[job_id] = workers
        
        # DO NOT WAIT FOR WORKERS - Let them run in background
        logger.info(f"✅ Pipeline started for job {job_id} - workers running in background")
        
    except Exception as e:
        logger.error(f"Error starting pipeline for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mark job as failed
        analysis_results[job_id]['status'] = 'failed'
        analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
        analysis_results[job_id]['error'] = str(e)
        
        # Clean up
        if job_id in active_job_tasks:
            del active_job_tasks[job_id]

async def monitor_job_completion(job_id: str, urls: List[str], workers: List[asyncio.Task], 
                               queues: dict, analysis_results: dict, url_task: asyncio.Task):
    """Monitor job completion without blocking"""
    try:
        # CRITICAL: Wait for all URLs to be queued before we start checking for completion
        # This eliminates the race condition where URLs might be "missing" because they haven't been queued yet
        logger.info(f"📋 Completion monitor waiting for all URLs to be queued...")
        await url_task
        logger.info(f"✅ All URLs queued, starting completion monitoring")
        
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Check if job is cancelled
            if analysis_results[job_id]['status'] in ['cancelled', 'cancelling']:
                logger.info(f"🛑 Job {job_id} cancelled, stopping completion monitor")
                break
            
            # Check completion based on processed URLs
            total_processed = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
            all_urls_processed = total_processed >= len(urls)
            
            # Check if all work is done - IMPROVED LOGIC
            all_queues_empty = all(q.empty() for q in queues.values())
            
            # Get pipeline stages to check if any channels are still being processed
            pipeline_stages = analysis_results[job_id].get('pipeline_stages', {})
            channels_still_processing = (
                pipeline_stages.get('queued_for_discovery', 0) +
                pipeline_stages.get('discovering_videos', 0) +
                pipeline_stages.get('queued_for_controversy', 0) +
                pipeline_stages.get('screening_controversy', 0) +
                pipeline_stages.get('queued_for_transcripts', 0) +
                pipeline_stages.get('fetching_transcripts', 0) +
                pipeline_stages.get('queued_for_llm', 0) +
                pipeline_stages.get('llm_processing', 0) +
                pipeline_stages.get('queued_for_results', 0) +
                pipeline_stages.get('result_processing', 0)
            )
            
            # Job is complete when:
            # 1. All URLs are processed (in results or failed_urls)
            # 2. All queues are empty
            # 3. No channels are still in processing stages
            job_complete = all_urls_processed and all_queues_empty and channels_still_processing == 0
            
            # DEBUG: Log detailed completion status when we're close to completion
            if all_queues_empty and not job_complete:
                logger.warning(f"🔍 DEBUG: Queues empty but job not complete:")
                logger.warning(f"   └─ Total URLs: {len(urls)}")
                logger.warning(f"   └─ Results: {len(analysis_results[job_id]['results'])}")
                logger.warning(f"   └─ Failed URLs: {len(analysis_results[job_id]['failed_urls'])}")
                logger.warning(f"   └─ Total processed: {total_processed}")
                logger.warning(f"   └─ All URLs processed: {all_urls_processed}")
                logger.warning(f"   └─ Channels still processing: {channels_still_processing}")
                logger.warning(f"   └─ Pipeline stages: {pipeline_stages}")
                
                # If all URLs are processed but we still have channels in processing stages,
                # this indicates a pipeline tracking issue
                if all_urls_processed and channels_still_processing > 0:
                    logger.warning(f"   └─ PIPELINE TRACKING ISSUE: All URLs processed but {channels_still_processing} channels still in processing stages")
                    logger.warning(f"   └─ This may indicate a race condition or tracking bug")
                    # Continue without sleeping - the pipeline stages should resolve themselves
                    continue
                
                # Check for missing URLs only if we don't have all URLs processed
                if not all_urls_processed:
                    # Since we waited for url_task, all URLs are definitely in the pipeline
                    # But we should still check if channels are processing to avoid false positives
                    if channels_still_processing > 0:
                        # Channels are still being processed, so "missing" URLs might just be in transit
                        continue
                    
                    # If we still have missing URLs at this point with no channels processing, it's a real issue
                    processed_urls = set(analysis_results[job_id]['results'].keys()) | set(f['url'] for f in analysis_results[job_id]['failed_urls'])
                    missing_urls = set(urls) - processed_urls
                    if missing_urls:
                        logger.warning(f"   └─ Missing URLs: {missing_urls}")
                        logger.info(f"🔍 MISSING_URLS DEBUG: Current failed_urls: {[f['url'] for f in analysis_results[job_id]['failed_urls']]}")
                        logger.info(f"🔍 MISSING_URLS DEBUG: Current results: {list(analysis_results[job_id]['results'].keys())}")
                        
                        # Check if missing URLs have controversy check failures
                        controversy_failures = analysis_results[job_id].get('controversy_check_failures', {})
                        for missing_url in missing_urls:
                            # CRITICAL FIX: Check if URL is already in failed_urls to avoid duplicates
                            already_in_failed = any(f['url'] == missing_url for f in analysis_results[job_id]['failed_urls'])
                            if already_in_failed:
                                logger.info(f"📋 URL {missing_url} already in failed_urls - skipping duplicate addition")
                                continue
                            
                            if missing_url in controversy_failures:
                                controversy_info = controversy_failures[missing_url]
                                logger.warning(f"   └─ Missing URL {missing_url} has controversy check failure: {controversy_info}")
                                
                                # Determine the appropriate error message and type based on controversy status
                                controversy_status = controversy_info.get('status', 'unknown')
                                controversy_reason = controversy_info.get('reason', 'Unknown reason')
                                channel_name = controversy_info.get('channel_name', 'Unknown')
                                
                                if controversy_status == 'controversial':
                                    # Controversial channels should already be in results, not failed_urls
                                    # Check if it's already in results
                                    if missing_url in analysis_results[job_id]['results']:
                                        logger.info(f"📋 Controversial channel {missing_url} already in results - no action needed")
                                        continue
                                    else:
                                        # This shouldn't happen with the new logic, but handle it just in case
                                        logger.warning(f"📋 Controversial channel {missing_url} missing from results - this is unexpected")
                                        error_msg = f"Channel flagged for controversy: {controversy_reason}"
                                        error_type = 'controversy_flagged'
                                elif controversy_status == 'not_controversial':
                                    # Channel passed controversy but had no successful videos
                                    error_msg = f"Channel processing completed but no videos were successfully analyzed. All video transcripts may have failed."
                                    error_type = 'no_videos_completed'
                                    logger.info(f"📋 Channel {missing_url} completed but no videos succeeded")
                                else:
                                    # Controversy screening had an error
                                    error_msg = f"Controversy screening failed: {controversy_reason}. Channel processing was incomplete."
                                    error_type = 'controversy_screening_error'
                                    logger.info(f"📋 Channel {missing_url} had controversy screening error")
                                
                                # Only add to failed_urls if not controversial (controversial ones are in results now)
                                if controversy_status != 'controversial':
                                    analysis_results[job_id]['failed_urls'].append({
                                        'url': missing_url,
                                        'error': error_msg,
                                        'error_type': error_type,
                                        'channel_name': channel_name,
                                        'controversy_status': controversy_status,
                                        'controversy_reason': controversy_reason
                                    })
                                    logger.info(f"✅ Added missing URL {missing_url} to failed_urls for proper completion tracking")
                                    logger.info(f"🔍 MISSING_URLS DEBUG: Added {missing_url} to failed_urls ({error_type}) - total failed: {len(analysis_results[job_id]['failed_urls'])}")
                            else:
                                logger.warning(f"   └─ Missing URL {missing_url} has no controversy check failure")
                                
                                # CRITICAL FIX: Check if URL is already in failed_urls to avoid duplicates
                                already_in_failed = any(f['url'] == missing_url for f in analysis_results[job_id]['failed_urls'])
                                if already_in_failed:
                                    logger.info(f"📋 URL {missing_url} already in failed_urls - skipping duplicate addition")
                                    continue
                                
                                # This shouldn't happen, but if it does, add it as a generic failure
                                analysis_results[job_id]['failed_urls'].append({
                                    'url': missing_url,
                                    'error': f"Channel was not properly tracked through the processing pipeline.",
                                    'error_type': 'pipeline_tracking_error',
                                    'channel_name': 'Unknown'
                                })
                                logger.info(f"✅ Added untracked URL {missing_url} to failed_urls")
                                logger.info(f"🔍 MISSING_URLS DEBUG: Added {missing_url} to failed_urls (pipeline_tracking_error) - total failed: {len(analysis_results[job_id]['failed_urls'])}")
                        
                        # Update total processed count after adding missing URLs
                        total_processed = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
                        all_urls_processed = total_processed >= len(urls)
                        logger.info(f"📊 Updated completion status: {total_processed}/{len(urls)} URLs processed")
            
            # Check for job completion with improved logic
            if job_complete:
                logger.info(f"✅ Job {job_id} completed - all URLs processed and no channels still processing")
                
                # Mark job as completed
                analysis_results[job_id]['status'] = 'completed'
                analysis_results[job_id]['completed_at'] = datetime.now().isoformat()
                analysis_results[job_id]['processed_urls'] = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
                
                # Create channel summaries
                await create_channel_summaries(job_id, analysis_results)
                
                # Cancel all workers
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                
                # Clean up task tracking
                if job_id in active_job_tasks:
                    del active_job_tasks[job_id]
                
                # Start next job in queue (this will handle current_job_id reset)
                await start_next_job(analysis_results)
                break
                
    except Exception as e:
        logger.error(f"Error in completion monitor for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())

async def update_queue_wait_times(analysis_results: dict):
    """Update estimated wait times for all queued jobs based on current progress"""
    async with job_queue_lock:
        global current_job_id
        
        if not job_queue or not current_job_id or current_job_id not in analysis_results:
            return
        
        current_job = analysis_results[current_job_id]
        
        # Calculate current job's remaining time
        current_progress = current_job['processed_urls'] / current_job['total_urls'] if current_job['total_urls'] > 0 else 0
        current_remaining_urls = current_job['total_urls'] - current_job['processed_urls']
        
        # Estimate time per URL based on current job performance
        elapsed_time = time.time() - current_job['performance_stats']['overall_start_time']
        if current_job['processed_urls'] > 0:
            time_per_url = elapsed_time / current_job['processed_urls']
        else:
            time_per_url = 30  # Default estimate: 30 seconds per URL
        
        # Calculate remaining time for current job
        current_job_remaining_minutes = (current_remaining_urls * time_per_url) / 60
        
        # Update wait times for all queued jobs
        for i, queued_job in enumerate(job_queue):
            queued_job_id = queued_job['job_id']
            
            # Calculate time for jobs ahead in queue
            jobs_ahead_time = 0
            for j in range(i):
                ahead_job = job_queue[j]
                jobs_ahead_time += (len(ahead_job['urls']) * time_per_url) / 60
            
            # Total wait time = current job remaining + jobs ahead
            total_wait_minutes = current_job_remaining_minutes + jobs_ahead_time
            
            # Update the job's wait time estimates
            analysis_results[queued_job_id]['estimated_wait_minutes'] = max(1, int(total_wait_minutes))
            analysis_results[queued_job_id]['estimated_start_time'] = (
                datetime.now() + timedelta(minutes=total_wait_minutes)
            ).isoformat()

async def start_next_job(analysis_results: dict):
    """Start the next job in queue if any"""
    async with job_queue_lock:
        global current_job_id
        
        # DEBUG: Log queue state when starting next job
        logger.info(f"🔍 START_NEXT_JOB DEBUG: current_job_id = {current_job_id}")
        logger.info(f"🔍 START_NEXT_JOB DEBUG: job_queue length = {len(job_queue)}")
        logger.info(f"🔍 START_NEXT_JOB DEBUG: active_job_tasks keys = {list(active_job_tasks.keys())}")
        
        if job_queue:
            next_job = job_queue.pop(0)
            current_job_id = next_job['job_id']  # CRITICAL FIX: Set current_job_id to the next job
            
            # Update status and reset queue position
            analysis_results[current_job_id]['status'] = 'processing'
            analysis_results[current_job_id]['queue_position'] = 0
            analysis_results[current_job_id]['estimated_wait_minutes'] = 0
            analysis_results[current_job_id]['estimated_start_time'] = None
            
            # Update queue positions for remaining jobs
            for i, queued_job in enumerate(job_queue):
                queued_job_id = queued_job['job_id']
                analysis_results[queued_job_id]['queue_position'] = i + 1
            
            logger.info(f"🚀 Starting next queued job {current_job_id}")
            
            # Start processing and track the task properly
            task = asyncio.create_task(
                process_creators_pipeline(
                    current_job_id,
                    next_job['urls'],
                    next_job['video_limit'],
                    next_job['llm_provider'],
                    analysis_results
                )
            )
            
            # Track task for cancellation - CRITICAL FIX
            active_job_tasks[current_job_id] = [task]
        else:
            # No more jobs in queue, reset current_job_id
            current_job_id = None
            logger.info("📋 No more jobs in queue, ready for new jobs")

async def create_channel_summaries(job_id: str, analysis_results: dict):
    """Create channel summaries from individual video results"""
    try:
        logger.info(f"📊 Creating channel summaries for job {job_id}")
        
        # Get categories for summary
        src_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(src_dir)
        categories_file = os.path.join(project_root, "data", "YouTube_Controversy_Categories.csv")
        categories_df = pd.read_csv(categories_file)
        all_categories = categories_df['Category'].tolist()
        
        # First, check for controversy failures that didn't make it to results
        controversy_failures = analysis_results[job_id].get('controversy_check_failures', {})
        for url, controversy_info in controversy_failures.items():
            # Check if this URL made it to results
            if url not in analysis_results[job_id]['results']:
                # Only add to failed_urls if it's NOT a controversial channel
                # Controversial channels should already be in results with proper scoring
                controversy_status = controversy_info.get('status', 'unknown')
                if controversy_status != 'controversial':
                    logger.warning(f"⚠️ Non-controversial channel {url} has no results - adding to failed URLs")
                    
                    # Check if it's already in failed_urls to avoid duplicates
                    already_failed = any(f['url'] == url for f in analysis_results[job_id]['failed_urls'])
                    if not already_failed:
                        analysis_results[job_id]['failed_urls'].append({
                            'url': url,
                            'error': f"Channel processing failed: {controversy_info.get('reason', 'Unknown reason')}",
                            'error_type': 'processing_failed',
                            'channel_name': controversy_info.get('channel_name', 'Unknown'),
                            'controversy_reason': controversy_info.get('reason', 'Unknown reason')
                        })
                else:
                    logger.info(f"📋 Controversial channel {url} should already be in results - skipping failed_urls addition")
        
        # Process each channel's video results
        channels_to_remove = []
        
        for url, channel_data in analysis_results[job_id]['results'].items():
            # Check if this channel has any successfully analyzed videos
            video_analyses = channel_data.get('video_analyses', [])
            
            if not video_analyses:
                # No videos were successfully analyzed - this is a failure
                logger.warning(f"❌ Channel {url} has no successfully analyzed videos - moving to failed")
                
                # Check if this was a controversial channel
                is_controversial = channel_data.get('controversy_flagged', False)
                controversy_reason = channel_data.get('controversy_reason', None)
                
                if is_controversial:
                    # Controversial channels should stay in results with their controversy scoring
                    # Don't move them to failed_urls - they already have proper analysis structure
                    logger.info(f"📋 Controversial channel {url} staying in results (has controversy analysis)")
                    continue
                else:
                    # Non-controversial channels with no videos should be moved to failed
                    error_msg = 'Failed to analyze any videos from this channel. All transcript downloads may have failed.'
                    error_type = 'no_videos_analyzed'
                    
                    # Add to failed URLs
                    analysis_results[job_id]['failed_urls'].append({
                        'url': url,
                        'error': error_msg,
                        'error_type': error_type,
                        'channel_name': channel_data.get('channel_name', 'Unknown'),
                        'channel_id': channel_data.get('channel_id', 'unknown'),
                        'video_count': len(channel_data.get('original_videos', [])),
                        'controversy_flagged': is_controversial,
                        'controversy_reason': controversy_reason
                    })
                    
                    # Mark for removal from results
                    channels_to_remove.append(url)
                    continue
                
            # Create summary across all videos for this channel
            summary = {}
            for category in all_categories:
                category_violations = []
                max_score = 0
                total_score = 0
                videos_with_violations = 0
                
                for video_analysis in video_analyses:
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
                        "total_videos": len(video_analyses),
                        "examples": sorted(category_violations, key=lambda x: x["score"], reverse=True)[:5]
                    }
            
            # Update the channel data with summary
            analysis_results[job_id]['results'][url]['summary'] = summary
        
        # Remove failed channels from results
        for url in channels_to_remove:
            del analysis_results[job_id]['results'][url]
            
        # Update final processed count (count channels, not videos)
        analysis_results[job_id]['processed_urls'] = len(analysis_results[job_id]['results']) + len(analysis_results[job_id]['failed_urls'])
        
        logger.info(f"✅ Created summaries for {len(analysis_results[job_id]['results'])} channels")
        if channels_to_remove:
            logger.info(f"❌ Moved {len(channels_to_remove)} channels to failed (no videos analyzed)")
        
    except Exception as e:
        logger.error(f"💥 Error creating channel summaries: {str(e)}")

async def process_job_with_cleanup(job_id: str, urls: List[str], video_limit: int, llm_provider: str, analysis_results: dict):
    """Process a job without blocking - just start the pipeline"""
    try:
        # Start the pipeline - this will return immediately
        await process_creators_pipeline(job_id, urls, video_limit, llm_provider, analysis_results)
    except Exception as e:
        logger.error(f"Error starting job {job_id}: {str(e)}")
        # If we fail to start, clean up and start next job
        await start_next_job(analysis_results) 